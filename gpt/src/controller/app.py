import gradio as gr
import whisper
import os
import openai

model = whisper.load_model("tiny")

# Load Environment Variables.
openai.api_key = os.getenv("OPENAI_API_KEY")
completion = openai.Completion()

def summerizeTransformerHandler(prompt, chat_log=None):
    """
    REST Request to the OpenAI API with a specific prompt
    :return: Response object with the status code 200 containing the result as String
    """
    start_sequence = "\nText:"
    examplePrompt="Write a list of the most important key takeaways from the text below. \n\nText:",
    
    prompt_text = f'{examplePrompt}{start_sequence}: {prompt}'
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=prompt_text,
      temperature=0.5,
      max_tokens=150,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      stop=[" Text:"],
    )
    gptResponse = response['choices'][0]['text']
    return str(gptResponse)

def whisperInference(audio):
    """
    Wishper Model inference to transform speech to text for a specific String
    :return: probs object with the result as String
    """
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)
    
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)

    return result.text

def whisperInferenceLanguage(audio):
    """
    Wishper Model inference to transform speech to language code for a specific String
    :return: probs object with the result as String
    """
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)
    
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    _, probs = model.detect_language(mel)

    return str({max(probs, key=probs.get)})


css = """
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: black;
            background: black;
        }
        input[type='range'] {
            accent-color: black;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 730px;
            margin: auto;
            padding-top: 1.5rem;
        }
     
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .prompt h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
"""

block = gr.Blocks(css=css)

with block:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 650px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <svg
                  width="0.65em"
                  height="0.65em"
                  viewBox="0 0 115 115"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <rect width="23" height="23" fill="white"></rect>
                  <rect y="69" width="23" height="23" fill="white"></rect>
                  <rect x="23" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="46" width="23" height="23" fill="white"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" width="23" height="23" fill="black"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="92" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="115" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="46" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="46" y="46" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="115" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="23" y="46" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="23" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="black"></rect>
                </svg>
                <h1 style="font-weight: 900; margin-bottom: 7px;">
                  PwC Tech Talk Speech Recognizer meets GPT
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
                Whisper is a general-purpose speech recognition model that is open source.
              </p>
              <p style="margin-bottom: 10px; font-size: 94%">
                When using the speech-to-text function only, no data is sent to third-party providers.
              </p>
            </div>
        """
    )

    with gr.Group():
        with gr.Box():
            with gr.Row().style(mobile_collapse=False, equal_height=True):
                audio = gr.Audio(
                    label="Input Audio",
                    show_label=False,
                    source="microphone",
                    type="filepath",
                    equal_height=True
                )

            with gr.Row().style(mobile_collapse=False, equal_height=True):
                transcribeButton = gr.Button("Transcribe Audio")
                langueageDetectionButton = gr.Button("Language Detection")  

            whisperOutputText = gr.Textbox(show_label=False)          
            whisperLanguageOutputText = gr.Textbox(show_label=False)
            
            with gr.Row().style(mobile_collapse=False, equal_height=True):
                gptSummarizeButton = gr.Button("GPT 3 Summarize Text")  

            gptSummorizeOutputText = gr.Textbox(show_label=False)

        transcribeButton.click(whisperInference, inputs=[audio], outputs=[whisperOutputText])
        langueageDetectionButton.click(whisperInferenceLanguage, inputs=[audio], outputs=[whisperLanguageOutputText])       
        gptSummarizeButton.click(summerizeTransformerHandler,inputs=[whisperOutputText], outputs=[gptSummorizeOutputText])


block.launch(server_name="0.0.0.0", server_port=7000)