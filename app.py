from __future__ import annotations

import base64
import io
import json
from typing import TYPE_CHECKING, TypedDict

import pdfplumber
import streamlit as st
from code_editor import code_editor as st_code_editor
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage

# from genson import SchemaBuilder
from PIL import Image

from settings import settings

# from pydantic import BaseModel

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile

st.set_page_config(layout="wide")

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


@st.cache_data
def extract_text(file: "UploadedFile") -> list[PageObject]:
    pages = []
    if file.type == "application/pdf":
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
    else:
        pages.append(
            PageObject(
                text="",
                image=get_b64_image(
                    Image.open(file.getvalue()),
                ),
            )
        )
    return pages


@st.cache_resource
def _llm():
    return ChatBedrockConverse(
        model=settings.LLM,
        region_name=settings.AWS_REGION,
        credentials_profile_name=settings.AWS_PROFILE,
        temperature=0,
    )


@st.cache_data
def _call_llm(pages: list[PageObject], schema: str | dict) -> dict:
    if not schema:
        raise ValueError("Specify a JSON schema")
    if isinstance(schema, str):
        schema = json.loads(schema)
    llm = _llm().with_structured_output(schema)
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


file_col, schema_col = st.columns(2)

with file_col:
    st.markdown("Document")
    uploaded_file = st.file_uploader(
        "File",
        type=["pdf", "jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="file_uploader",
    )

    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            st.pdf(uploaded_file)
        else:
            st.image(uploaded_file)

with schema_col:
    st.markdown(
        "JSON Schema ([generate from json](https://transform.tools/json-to-json-schema))"
    )
    code_editor = st_code_editor(
        "{}", lang="json", allow_reset=True, response_mode="blur", key="code_editor"
    )

if st.button("Run", key="run"):
    pages = extract_text(uploaded_file)
    schema = code_editor.get("text", "")
    with st.spinner("Calling LLM..."):
        response = _call_llm(pages, schema)
    tabs = st.tabs(["JSON", "Text"])
    with tabs[0]:
        st.json(response)
    with tabs[1]:
        st.code(json.dumps(response, indent=2), language="json")


# if json_input.get("type", "") == "submit":
#     builder = SchemaBuilder()
#     builder.add_object(json.loads(json_input.get("text", "")))
#     model_py = builder.to_json(indent=2)

#     with output_col:
#         st.code(str(model_py), language="python")

#     class Model(BaseModel):
#         pass

#     exec(model_py)
#     st.json(Model.model_json_schema())  # type: ignore # noqa: F821
