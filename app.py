import base64
import io
import json
from typing import TYPE_CHECKING, TypedDict

import pdfplumber
import streamlit as st
from code_editor import code_editor as st_code_editor
from genson import SchemaBuilder
from PIL import Image
from pydantic import BaseModel

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile

st.set_page_config(layout="wide")


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


def extract_text(file: "UploadedFile") -> list[PageObject]:
    pages = []
    if file.type == "application/pdf":
        pdf: pdfplumber.PDF = pdfplumber.open(file.getvalue())
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
    json_input = st_code_editor("{}", lang="json", allow_reset=True, key="code_editor")

if uploaded_file and json_input.get("type", "") == "submit":
    # extract text from doc
    pages = extract_text(uploaded_file)
    # call llm with struvtured output
    pass


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
