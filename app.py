import os
os.system("pip uninstall -y gradio")
os.system("pip install gradio==3.43.0")
from lmdeploy.serve.gradio.turbomind_coupled import *
from lmdeploy.messages import TurbomindEngineConfig
from lmdeploy import ChatTemplateConfig

chat_template = ChatTemplateConfig(model_name='internlm-chat-7b', system='', eosys='', meta_instruction='')
backend_config = TurbomindEngineConfig(model_name='internlm-chat-7b', max_batch_size=1, cache_max_entry_count=0.05)#, model_format='awq')
model_path = 'zzfive/DAIGT-InternLM-small'

InterFace.async_engine = AsyncEngine(
    model_path=model_path,
    backend='turbomind',
    backend_config=backend_config,
    chat_template_config=chat_template,
    tp=1)


async def reset_local_func(instruction_txtbox: gr.Textbox,
                           state_chatbot: Sequence, session_id: int):
    """reset the session.
    Args:
        instruction_txtbox (str): user's prompt
        state_chatbot (Sequence): the chatting history
        session_id (int): the session id
    """
    state_chatbot = []
    # end the session
    with InterFace.lock:
        InterFace.global_session_id += 1
        session_id = InterFace.global_session_id
    return (state_chatbot, state_chatbot, gr.Textbox.update(value=''), session_id)


async def cancel_local_func(state_chatbot: Sequence, cancel_btn: gr.Button,
                            reset_btn: gr.Button, session_id: int):
    """stop the session.
    Args:
        instruction_txtbox (str): user's prompt
        state_chatbot (Sequence): the chatting history
        cancel_btn (gr.Button): the cancel button
        reset_btn (gr.Button): the reset button
        session_id (int): the session id
    """
    yield (state_chatbot, disable_btn, disable_btn, session_id)
    InterFace.async_engine.stop_session(session_id)
    # pytorch backend does not support resume chat history now
    if InterFace.async_engine.backend == 'pytorch':
        yield (state_chatbot, disable_btn, enable_btn, session_id)
    else:
        with InterFace.lock:
            InterFace.global_session_id += 1
            session_id = InterFace.global_session_id
        messages = []
        for qa in state_chatbot:
            messages.append(dict(role='user', content=qa[0]))
            if qa[1] is not None:
                messages.append(dict(role='assistant', content=qa[1]))
        gen_config = GenerationConfig(max_new_tokens=0)
        async for out in InterFace.async_engine.generate(messages,
                                                         session_id,
                                                         gen_config=gen_config,
                                                         stream_response=True,
                                                         sequence_start=True,
                                                         sequence_end=False):
            pass
        yield (state_chatbot, disable_btn, enable_btn, session_id)


with gr.Blocks(css=CSS, theme=THEME) as demo:
    state_chatbot = gr.State([])
    state_session_id = gr.State(0)

    with gr.Column(elem_id='container'):
        gr.Markdown('## LMDeploy Playground')
        gr.Markdown('[InternLM Math GitHub Page](https://github.com/InternLM/InternLM-Math)')

        chatbot = gr.Chatbot(
            elem_id='chatbot',
            label=InterFace.async_engine.engine.model_name)
        instruction_txtbox = gr.Textbox(
            placeholder='Please input the instruction',
            label='Instruction')
        with gr.Row():
            cancel_btn = gr.Button(value='Cancel', interactive=False)
            reset_btn = gr.Button(value='Reset')
        with gr.Row():
            request_output_len = gr.Slider(1,
                                            2048,
                                            value=1024,
                                            step=1,
                                            label='Maximum new tokens')
            top_p = gr.Slider(0.01, 1, value=1.0, step=0.01, label='Top_p')
            temperature = gr.Slider(0.01,
                                    1.5,
                                    value=0.01,
                                    step=0.01,
                                    label='Temperature')

    send_event = instruction_txtbox.submit(chat_stream_local, [
        instruction_txtbox, state_chatbot, cancel_btn, reset_btn,
        state_session_id, top_p, temperature, request_output_len
    ], [state_chatbot, chatbot, cancel_btn, reset_btn])
    instruction_txtbox.submit(
        lambda: gr.Textbox.update(value=''),
        [],
        [instruction_txtbox],
    )
    cancel_btn.click(
        cancel_local_func,
        [state_chatbot, cancel_btn, reset_btn, state_session_id],
        [state_chatbot, cancel_btn, reset_btn, state_session_id],
        cancels=[send_event])

    reset_btn.click(reset_local_func,
                    [instruction_txtbox, state_chatbot, state_session_id],
                    [state_chatbot, chatbot, instruction_txtbox, state_session_id],
                    cancels=[send_event])

    def init():
        with InterFace.lock:
            InterFace.global_session_id += 1
        new_session_id = InterFace.global_session_id
        return new_session_id

    demo.load(init, inputs=None, outputs=[state_session_id])

# demo.queue(concurrency_count=InterFace.async_engine.instance_num,
            # max_size=100).launch()
demo.queue(max_size=1000).launch(max_threads=InterFace.async_engine.instance_num)