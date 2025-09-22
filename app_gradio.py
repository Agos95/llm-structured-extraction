from __future__ import annotations

import base64
import io
import json
from typing import TYPE_CHECKING, TypedDict

import gradio as gr
import pdfplumber
from gradio_pdf import PDF as gr_pdf
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image

from settings import settings

_PROMPT = """\
You are a powerful json extractor.
You will be provided with a set of images and optionally the corresponding OCR text for each page of the document. 
Your task is to extract all the relevant information following the provided json schema.

For each page of the document, analyze both the image and the OCR text if present to accurately extract the required information. Combine the text and visual cues to ensure consistency and completeness of the extracted data. The extracted text may not contain all the information: some details may be only present in the image.

Once you have extracted all the necessary information, generate a well-structured JSON object that represents the invoice document data. 
The JSON object should be valid and ready to be stored in a database without further processing.

Consider the following additional points:
- Handle cases where information may be spread across multiple pages or sections.
- Validate and clean the extracted data to ensure accuracy and consistency.
- If any required information is missing or unclear, indicate it in the JSON output with a value of null.
- Use appropriate data types and structures within the JSON object.
- Ensure proper escaping of special characters and handling of unicode characters in the JSON output.
- For date objects use the format 'YYYY-MM-DD'
- Ensure to extract the data from each page. Do not skip any information from the pages.
"""


class PageObject(TypedDict):
    text: str
    image: str


def get_b64_image(image: Image) -> str:
    """Image to BS64 converted

    Args:
        image (Image): the image to be converted

    Returns:
        str: BS64 image string
    """
    buff = io.BytesIO()
    image.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return img_str


def extract_text(file: str) -> list[PageObject]:
    pages = []
    pdf: pdfplumber.PDF = pdfplumber.open(file)
    for i, page in enumerate(pdf.pages, start=1):
        pages.append(
            PageObject(
                text=f"""\
                ########## Page {i} ##########
                
                {page.extract_text(layout=True, use_text_flow=False)}

                ##############################
                """,
                image=get_b64_image(page.to_image(resolution=300)),
            )
        )
    # else:
    #     pages.append(
    #         PageObject(
    #             text="",
    #             image=get_b64_image(
    #                 Image.open(file.getvalue()),
    #             ),
    #         )
    #     )
    return pages


_llm = ChatBedrockConverse(
    model=settings.LLM,
    region_name=settings.AWS_REGION,
    credentials_profile_name=settings.AWS_PROFILE,
    temperature=0,
)


def _call_llm(pages: list[PageObject], schema: str | dict) -> dict:
    if not schema:
        raise ValueError("Specify a JSON schema")
    if isinstance(schema, str):
        schema = json.loads(schema)
    llm = _llm.with_structured_output(schema)
    messages = [SystemMessage(_PROMPT)]

    for page in pages:
        if page["text"]:
            messages.append(
                HumanMessage(
                    [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": page["image"],
                            },
                        },
                        {
                            "type": "text",
                            "text": page["text"],
                        },
                    ]
                )
            )
        else:
            messages.append(
                HumanMessage(
                    [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": page["image"],
                            },
                        },
                    ]
                )
            )
    response = llm.invoke(messages)
    return response


def _load_file(path: str):
    return gr_pdf(path, visible=True)


def _run(path: str, schema: str):
    pages = extract_text(path)
    response = _call_llm(pages, schema)

    return gr.Code(json.dumps(response, indent=2), visible=True)


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("Document")
            uploaded_file = gr.File(file_types=[".pdf"])
            pdf = gr_pdf(visible=False)

            uploaded_file.upload(_load_file, inputs=uploaded_file, outputs=pdf)

        with gr.Column():
            gr.Markdown(
                "JSON Schema ([generate from json](https://transform.tools/json-to-json-schema))"
            )
            code_editor = gr.Code("", language="json", interactive=True)

    button = gr.Button("Run")
    structured_output = gr.Code(language="json", visible=False)

    button.click(_run, inputs=[uploaded_file, code_editor], outputs=structured_output)


if __name__ == "__main__":
    demo.launch()
