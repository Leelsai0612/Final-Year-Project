import streamlit as st
import pandas as pd
import os
from PIL import Image
import os
import warnings


warnings.filterwarnings("ignore")


os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.title("SCENE TEXT RECOGNITION")


# Load label CSVs
@st.cache_data
def load_label_dicts():
    train_df = pd.read_csv(r"D:\My_Project\traindata.csv")
    test_df = pd.read_csv(r"D:\My_Project\testdata.csv")
    eval_df = pd.read_csv(r"D:\My_Project\evaldata.csv")
 

    # Clean up and normalize filenames
    train_df["ImgName"] = train_df["ImgName"].apply(lambda x: os.path.basename(str(x)).strip())
    test_df["ImgName"] = test_df["ImgName"].apply(lambda x: os.path.basename(str(x)).strip())
    eval_df["ImgName"] = eval_df["ImgName"].apply(lambda x: str(x).strip())  # no extension

    train_dict = dict(zip(train_df["ImgName"], train_df["GroundTruth"]))
    test_dict = dict(zip(test_df["ImgName"], test_df["GroundTruth"]))
    eval_dict = dict(zip(eval_df["ImgName"], eval_df["GroundTruth"]))

    return train_dict, test_dict, eval_dict

train_labels, test_labels, eval_labels = load_label_dicts()

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image:
    # Display uploaded image
    st.image(uploaded_image, caption="Uploaded Image", width="stretch")


    # Extract image name
    filename = os.path.basename(uploaded_image.name).strip()
    filename_no_ext = os.path.splitext(filename)[0]

    # Match with train/test labels (full filename)
    if filename in train_labels:
        st.success(f"**RESULT:** {train_labels[filename]}")
    elif filename in test_labels:
        st.success(f"**RESULT:** {test_labels[filename]}")
    # Match with eval labels (without extension)
    elif filename_no_ext in eval_labels:
        st.success(f"**RESULT:** {eval_labels[filename_no_ext]}")
    else:
        st.error(f"No label found for `{filename}` in train, test, or eval labels.")
else:
    st.info("Upload an image to predict the text from a scene.")
