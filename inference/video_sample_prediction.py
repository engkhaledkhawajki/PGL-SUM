import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import cv2
from tqdm import tqdm
import os
import torch
import numpy as np
from layers.summarizer import PGL_SUM
import json
import argparse


class VideoSamplePrediction:
    """
    This class wraps the loading of a video, breaking it down into frames and applying
    features extraction for its frames
    and return a numpy array of shape (n_frames, feature_size)
    Initially the feature extractor is set fixed to DenseNet121. (later could make it passed by the user)
    """

    def __init__(self, video_path, output_folder_path):
        self.video_path = video_path
        self.video_data_path = None
        self.video_frames_path = None
        self._init_output_folders(output_folder_path)

        self.feature_extractor = models.densenet121(pretrained=True)
        self.feature_extractor.classifier = nn.Linear(in_features=1024, out_features=1024, bias=True)
        self.feature_extractor.eval()

    def _init_output_folders(self, output_folder_path):
        if output_folder_path != "":
            video_data_path = output_folder_path
            video_frames_path = video_data_path + "/video_frames"
        else:
            current_dir = os.getcwd()
            video_data_path = "/video_data"
            video_data_path = os.path.join(current_dir, video_data_path)
            video_frames_path = video_data_path + "/video_frames"
            video_frames_path = os.path.join(current_dir, video_frames_path)

        if not os.path.exists(video_data_path):
            os.makedirs(video_data_path)

        if not os.path.exists(video_frames_path):
            os.makedirs(video_frames_path)

        self.video_data_path = video_data_path
        self.video_frames_path = video_frames_path

    def extract_video_frames(self):

        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        progress_bar = tqdm(total=total_frames,
                            desc='Extracting Frames',
                            position=0,
                            leave=True,
                            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for frame_number in range(total_frames):

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if ret:
                frame_path = os.path.join(self.video_frames_path, f"frame_{frame_number:05d}.jpg")
                cv2.imwrite(frame_path, frame)
                progress_bar.update(1)

        progress_bar.close()

        cap.release()

    def _load_video_frames(self):

        files = os.listdir(self.video_frames_path)
        image_files = [file for file in files if file.endswith('jpg')]

        images = []
        for file in image_files:
            file_path = os.path.join(self.video_frames_path, file)
            image = cv2.imread(file_path)
            images.append(image)
        return images

    def extract_video_features(self):

        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        video_features = []

        loaded_frames = self._load_video_frames()

        frame_count = 0
        total_frames = len(loaded_frames)

        progress_bar = tqdm(total=total_frames,
                            desc='Extracting Features',
                            position=0,
                            leave=True,
                            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for frame in loaded_frames:
            img = preprocess(frame)
            img = img.unsqueeze(0)

            with torch.no_grad():
                features = self.feature_extractor(img)

            video_features.append(features.squeeze().numpy())

            frame_count += 1
            progress_bar.update(1)

        progress_bar.close()

        np.save(f"{self.video_data_path}/video_features.npy", np.array(video_features))
        return np.array(video_features)

    def predict(self, video_features, model):
        model.eval()

        out_scores_dict = {}

        with torch.no_grad():
            scores, attn_weights = model(video_features)  # [1, seq_len]
            scores = scores.squeeze(0).cpu().numpy().tolist()

            out_scores_dict['video'] = scores

        scores_save_path = f"{self.video_data_path}/scores.json"
        with open(scores_save_path, 'w') as f:
            tqdm.write(f'Saving score at {str(scores_save_path)}.')
            json.dump(out_scores_dict, f)
        return scores


if __name__ == "__main__":
    # arguments to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, help="Path of the video to summarize.")
    parser.add_argument("--model_path", type=str, help="Path of the pretrained weights.")
    parser.add_argument("--output_folder_path", type=str, help="Path of the output folder (video frames and features).")

    args = vars(parser.parse_args())
    video_path = args["video_path"]
    model_path = args["model_path"]
    output_folder_path = args["output_folder_path"]

    video_sample_loader = VideoSamplePrediction(video_path=video_path, output_folder_path=output_folder_path or "")
    video_sample_loader.extract_video_frames()
    video_features = video_sample_loader.extract_video_features()

    trained_model = PGL_SUM(input_size=1024, output_size=1024, num_segments=4, heads=8,
                            fusion="add", pos_enc="absolute")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model.load_state_dict(torch.load(model_path, map_location=device))

    scores = video_sample_loader.predict(torch.tensor(video_features), trained_model)

    print(scores)

    # TODO:
    #  1. Generate Summary
