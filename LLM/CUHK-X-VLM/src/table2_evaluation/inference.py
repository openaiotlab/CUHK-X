import os
import pandas as pd
from openai import OpenAI
import base64
import random
import argparse


# convert to base64 encoding
def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode('utf-8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, help='API', required=True)
    parser.add_argument('--base_url', type=str, help='URL', required=True)
    parser.add_argument('--model', type=str, help='model type', default="doubao-seed-1-6-vision-250815")
    parser.add_argument('--file_path', type=str, help='file path', default="result.csv")
    args = parser.parse_args()
    print(args.api_key, args.base_url, args.model, args.file_path)
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )

    df = pd.read_csv(args.file_path, header=0)
    for index, row in df.iterrows():
        video_path = row["path"]
        print("video_path:" + video_path)

        try:
            base64_video = encode_video(video_path)
        except Exception as e:
            print(e)
            continue

        # create session
        completion = client.chat.completions.create(
            # Model ID
            model= args.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"data:video/mp4;base64,{base64_video}",
                            }
                        },
                        {
                            "type": "text",
                            "text": "视频里的人做了什么事？"
                        }
                    ],
                }
            ],
        )
        print("output:" + completion.choices[0].message.content)
        df.iloc[index, 1] = completion.choices[0].message.content
        df.to_csv(args.file_path, mode='w', header=True, index=False)
