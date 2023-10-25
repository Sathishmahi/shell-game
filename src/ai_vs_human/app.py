# ffmpeg -i artifact/ai_vs_human/output.mp4  -vcodec libx264 artifact/ai_vs_human/output1.mp4     display cv2.writer video commend

import streamlit as st
import os
from predict import Track
from utils import read_yaml

config_con = read_yaml()

artifact_con = config_con.get("artifact")
artifact_root_dir =  artifact_con.get("root_dir")
ai_vs_human_con = config_con.get("ai_vs_human")
ai_vs_human_root_dir = os.path.join(
    artifact_root_dir,
    ai_vs_human_con.get("root_dir"))

os.makedirs(ai_vs_human_root_dir,exist_ok=True)
input_file_path = os.path.join(ai_vs_human_root_dir,ai_vs_human_con.get("input_video_file_name"))
input_display_file_path = os.path.join(ai_vs_human_root_dir,ai_vs_human_con.get("input_dis_video_file_name"))


import subprocess

input_video = input_file_path
output_video = input_display_file_path

command = f"ffmpeg -i {input_video} -vcodec libx264 {output_video}"

try:
    subprocess.run(command, shell=True, check=True)
    print("Video conversion completed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Video conversion failed with error: {e}")

if os.path.exists(input_display_file_path):

    st.title("Shell Game AI vs Human")

    st.subheader("Input Video")

    st.video(input_display_file_path)

    st.write(" After Watch The Video Enter The Zone ID ")

    zone_id = st.text_input(label = f" Enter the Zone ID :) " )

    