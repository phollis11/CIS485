This document is designed to assist with the creation and implemetnation of the key point detection model. It provides clear examples of code that are used to create the key point detection model. Additionally, it provides examples of how to use homography within this system and how to implement the team classification system.

Helper functions - !pip install -q git+https://github.com/roboflow/sports.git

Player detection - import supervision as sv

SOURCE_VIDEO_PATH = "/content/121364_0.mp4"

box_annotator = sv.BoxAnnotator(
color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
thickness=2
)
label_annotator = sv.LabelAnnotator(
color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
text_color=sv.Color.from_hex('#000000')
)

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame = next(frame_generator)

result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
detections = sv.Detections.from_inference(result)

labels = [
f"{class_name} {confidence:.2f}"
for class_name, confidence
in zip(detections['class_name'], detections.confidence)
]

annotated_frame = frame.copy()
annotated_frame = box_annotator.annotate(
scene=annotated_frame,
detections=detections)
annotated_frame = label_annotator.annotate(
scene=annotated_frame,
detections=detections,
labels=labels)

sv.plot_image(annotated_frame)

player detection again - import supervision as sv

SOURCE_VIDEO_PATH = "/content/121364_0.mp4"
BALL_ID = 0

ellipse_annotator = sv.EllipseAnnotator(
color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
thickness=2
)
label_annotator = sv.LabelAnnotator(
color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
text_color=sv.Color.from_hex('#000000'),
text_position=sv.Position.BOTTOM_CENTER
)
triangle_annotator = sv.TriangleAnnotator(
color=sv.Color.from_hex('#FFD700'),
base=25,
height=21,
outline_thickness=1
)

tracker = sv.ByteTrack()
tracker.reset()

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame = next(frame_generator)

result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
detections = sv.Detections.from_inference(result)

ball_detections = detections[detections.class_id == BALL_ID]
ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

all_detections = detections[detections.class_id != BALL_ID]
all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
all_detections.class_id -= 1
all_detections = tracker.update_with_detections(detections=all_detections)

labels = [
f"#{tracker_id}"
for tracker_id
in all_detections.tracker_id
]

annotated_frame = frame.copy()
annotated_frame = ellipse_annotator.annotate(
scene=annotated_frame,
detections=all_detections)
annotated_frame = label_annotator.annotate(
scene=annotated_frame,
detections=all_detections,
labels=labels)
annotated_frame = triangle_annotator.annotate(
scene=annotated_frame,
detections=ball_detections)

sv.plot_image(annotated_frame)

Seperating teams - from tqdm import tqdm

SOURCE_VIDEO_PATH = "/content/121364_0.mp4"
PLAYER_ID = 2
STRIDE = 30

frame_generator = sv.get_video_frames_generator(
source_path=SOURCE_VIDEO_PATH, stride=STRIDE)

crops = []
for frame in tqdm(frame_generator, desc='collecting crops'):
result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
detections = sv.Detections.from_inference(result)
detections = detections.with_nms(threshold=0.5, class_agnostic=True)
detections = detections[detections.class_id == PLAYER_ID]
players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
crops += players_crops

    import numpy as np

from more_itertools import chunked

BATCH_SIZE = 32

crops = [sv.cv2_to_pillow(crop) for crop in crops]
batches = chunked(crops, BATCH_SIZE)
data = []
with torch.no_grad():
for batch in tqdm(batches, desc='embedding extraction'):
inputs = EMBEDDINGS_PROCESSOR(images=batch, return_tensors="pt").to(DEVICE)
outputs = EMBEDDINGS_MODEL(\*\*inputs)
embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
data.append(embeddings)

data = np.concatenate(data)

import umap
from sklearn.cluster import KMeans

REDUCER = umap.UMAP(n_components=3)
CLUSTERING_MODEL = KMeans(n_clusters=2)

projections = REDUCER.fit_transform(data)
clusters = CLUSTERING_MODEL.fit_predict(projections)

import supervision as sv
from tqdm import tqdm
from sports.common.team import TeamClassifier

SOURCE_VIDEO_PATH = "/content/121364_0.mp4"
PLAYER_ID = 2
STRIDE = 30

frame_generator = sv.get_video_frames_generator(
source_path=SOURCE_VIDEO_PATH, stride=STRIDE)

crops = []
for frame in tqdm(frame_generator, desc='collecting crops'):
result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
detections = sv.Detections.from_inference(result)
players_detections = detections[detections.class_id == PLAYER_ID]
players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
crops += players_crops

team_classifier = TeamClassifier(device="cuda")
team_classifier.fit(crops)

goalkeepers import numpy as np
import supervision as sv

def resolve_goalkeepers_team_id(
players: sv.Detections,
goalkeepers: sv.Detections
) -> np.ndarray:
goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)
goalkeepers_team_id = []
for goalkeeper_xy in goalkeepers_xy:
dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

    return np.array(goalkeepers_team_id)

    import supervision as sv

SOURCE_VIDEO_PATH = "/content/121364_0.mp4"
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

ellipse_annotator = sv.EllipseAnnotator(
color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
thickness=2
)
label_annotator = sv.LabelAnnotator(
color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
text_color=sv.Color.from_hex('#000000'),
text_position=sv.Position.BOTTOM_CENTER
)
triangle_annotator = sv.TriangleAnnotator(
color=sv.Color.from_hex('#FFD700'),
base=25,
height=21,
outline_thickness=1
)

tracker = sv.ByteTrack()
tracker.reset()

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame = next(frame_generator)

result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
detections = sv.Detections.from_inference(result)

ball_detections = detections[detections.class_id == BALL_ID]
ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

all_detections = detections[detections.class_id != BALL_ID]
all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
all_detections = tracker.update_with_detections(detections=all_detections)

goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
players_detections = all_detections[all_detections.class_id == PLAYER_ID]
referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
players_detections.class_id = team_classifier.predict(players_crops)

goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
players_detections, goalkeepers_detections)

referees_detections.class_id -= 1

all_detections = sv.Detections.merge([
players_detections, goalkeepers_detections, referees_detections])

labels = [
f"#{tracker_id}"
for tracker_id
in all_detections.tracker_id
]

all_detections.class_id = all_detections.class_id.astype(int)

annotated_frame = frame.copy()
annotated_frame = ellipse_annotator.annotate(
scene=annotated_frame,
detections=all_detections)
annotated_frame = label_annotator.annotate(
scene=annotated_frame,
detections=all_detections,
labels=labels)
annotated_frame = triangle_annotator.annotate(
scene=annotated_frame,
detections=ball_detections)

sv.plot_image(annotated_frame)

keypoint detection -

import supervision as sv

SOURCE_VIDEO_PATH = "/content/121364_0.mp4"

vertex_annotator = sv.VertexAnnotator(
color=sv.Color.from_hex('#FF1493'),
radius=8)

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame = next(frame_generator)

result = FIELD_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
key_points = sv.KeyPoints.from_inference(result)

annotated_frame = frame.copy()
annotated_frame = vertex_annotator.annotate(
scene=annotated_frame,
key_points=key_points)

sv.plot_image(annotated_frame)

filter import supervision as sv

SOURCE_VIDEO_PATH = "/content/121364_0.mp4"

vertex_annotator = sv.VertexAnnotator(
color=sv.Color.from_hex('#FF1493'),
radius=8)

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH, start=200)
frame = next(frame_generator)

result = FIELD_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
key_points = sv.KeyPoints.from_inference(result)

filter = key_points.confidence[0] > 0.5
frame_reference_points = key_points.xy[0][filter]
frame_reference_key_points = sv.KeyPoints(
xy=frame_reference_points[np.newaxis, ...])

annotated_frame = frame.copy()
annotated_frame = vertex_annotator.annotate(
scene=annotated_frame,
key_points=frame_reference_key_points)

sv.plot_image(annotated_frame)

from sports.annotators.soccer import draw_pitch
from sports.configs.soccer import SoccerPitchConfiguration

CONFIG = SoccerPitchConfiguration()

annotated_frame = draw_pitch(CONFIG)

sv.plot_image(annotated_frame)

import numpy as np
import supervision as sv
from sports.common.view import ViewTransformer

SOURCE_VIDEO_PATH = "/content/121364_0.mp4"

edge_annotator = sv.EdgeAnnotator(
color=sv.Color.from_hex('#00BFFF'),
thickness=2, edges=CONFIG.edges)
vertex_annotator = sv.VertexAnnotator(
color=sv.Color.from_hex('#FF1493'),
radius=8)
vertex_annotator_2 = sv.VertexAnnotator(
color=sv.Color.from_hex('#00BFFF'),
radius=8)

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH, start=200)
frame = next(frame_generator)

result = FIELD_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
key_points = sv.KeyPoints.from_inference(result)

filter = key_points.confidence[0] > 0.5
frame_reference_points = key_points.xy[0][filter]
frame_reference_key_points = sv.KeyPoints(
xy=frame_reference_points[np.newaxis, ...])

pitch_reference_points = np.array(CONFIG.vertices)[filter]

transformer = ViewTransformer(
source=pitch_reference_points,
target=frame_reference_points
)

pitch_all_points = np.array(CONFIG.vertices)
frame_all_points = transformer.transform_points(points=pitch_all_points)

frame_all_key_points = sv.KeyPoints(xy=frame_all_points[np.newaxis, ...])

annotated_frame = frame.copy()
annotated_frame = edge_annotator.annotate(
scene=annotated_frame,
key_points=frame_all_key_points)
annotated_frame = vertex_annotator_2.annotate(
scene=annotated_frame,
key_points=frame_all_key_points)
annotated_frame = vertex_annotator.annotate(
scene=annotated_frame,
key_points=frame_reference_key_points)

sv.plot_image(annotated_frame)

Final project - import supervision as sv
from sports.annotators.soccer import (
draw_pitch,
draw_points_on_pitch,
draw_pitch_voronoi_diagram
)

SOURCE_VIDEO_PATH = "/content/121364_0.mp4"
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

ellipse_annotator = sv.EllipseAnnotator(
color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
thickness=2
)
label_annotator = sv.LabelAnnotator(
color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
text_color=sv.Color.from_hex('#000000'),
text_position=sv.Position.BOTTOM_CENTER
)
triangle_annotator = sv.TriangleAnnotator(
color=sv.Color.from_hex('#FFD700'),
base=20, height=17
)

tracker = sv.ByteTrack()
tracker.reset()

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame = next(frame_generator)

# ball, goalkeeper, player, referee detection

result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
detections = sv.Detections.from_inference(result)

ball_detections = detections[detections.class_id == BALL_ID]
ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

all_detections = detections[detections.class_id != BALL_ID]
all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
all_detections = tracker.update_with_detections(detections=all_detections)

goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
players_detections = all_detections[all_detections.class_id == PLAYER_ID]
referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

# team assignment

players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
players_detections.class_id = team_classifier.predict(players_crops)

goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
players_detections, goalkeepers_detections)

referees_detections.class_id -= 1

all_detections = sv.Detections.merge([
players_detections, goalkeepers_detections, referees_detections])

# frame visualization

labels = [
f"#{tracker_id}"
for tracker_id
in all_detections.tracker_id
]

all_detections.class_id = all_detections.class_id.astype(int)

annotated_frame = frame.copy()
annotated_frame = ellipse_annotator.annotate(
scene=annotated_frame,
detections=all_detections)
annotated_frame = label_annotator.annotate(
scene=annotated_frame,
detections=all_detections,
labels=labels)
annotated_frame = triangle_annotator.annotate(
scene=annotated_frame,
detections=ball_detections)

sv.plot_image(annotated_frame)

players_detections = sv.Detections.merge([
players_detections, goalkeepers_detections
])

# detect pitch key points

result = FIELD_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
key_points = sv.KeyPoints.from_inference(result)

# project ball, players and referies on pitch

filter = key_points.confidence[0] > 0.5
frame_reference_points = key_points.xy[0][filter]
pitch_reference_points = np.array(CONFIG.vertices)[filter]

transformer = ViewTransformer(
source=frame_reference_points,
target=pitch_reference_points
)

frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
pitch_players_xy = transformer.transform_points(points=players_xy)

referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
pitch_referees_xy = transformer.transform_points(points=referees_xy)

# visualize video game-style radar view

annotated_frame = draw_pitch(CONFIG)
annotated_frame = draw_points_on_pitch(
config=CONFIG,
xy=pitch_ball_xy,
face_color=sv.Color.WHITE,
edge_color=sv.Color.BLACK,
radius=10,
pitch=annotated_frame)
annotated_frame = draw_points_on_pitch(
config=CONFIG,
xy=pitch_players_xy[players_detections.class_id == 0],
face_color=sv.Color.from_hex('00BFFF'),
edge_color=sv.Color.BLACK,
radius=16,
pitch=annotated_frame)
annotated_frame = draw_points_on_pitch(
config=CONFIG,
xy=pitch_players_xy[players_detections.class_id == 1],
face_color=sv.Color.from_hex('FF1493'),
edge_color=sv.Color.BLACK,
radius=16,
pitch=annotated_frame)
annotated_frame = draw_points_on_pitch(
config=CONFIG,
xy=pitch_referees_xy,
face_color=sv.Color.from_hex('FFD700'),
edge_color=sv.Color.BLACK,
radius=16,
pitch=annotated_frame)

sv.plot_image(annotated_frame)

# visualize voronoi diagram

annotated_frame = draw_pitch(CONFIG)
annotated_frame = draw_pitch_voronoi_diagram(
config=CONFIG,
team_1_xy=pitch_players_xy[players_detections.class_id == 0],
team_2_xy=pitch_players_xy[players_detections.class_id == 1],
team_1_color=sv.Color.from_hex('00BFFF'),
team_2_color=sv.Color.from_hex('FF1493'),
pitch=annotated_frame)

sv.plot_image(annotated_frame)

# visualize voronoi diagram with blend

annotated_frame = draw_pitch(
config=CONFIG,
background_color=sv.Color.WHITE,
line_color=sv.Color.BLACK
)
annotated_frame = draw_pitch_voronoi_diagram_2(
config=CONFIG,
team_1_xy=pitch_players_xy[players_detections.class_id == 0],
team_2_xy=pitch_players_xy[players_detections.class_id == 1],
team_1_color=sv.Color.from_hex('00BFFF'),
team_2_color=sv.Color.from_hex('FF1493'),
pitch=annotated_frame)
annotated_frame = draw_points_on_pitch(
config=CONFIG,
xy=pitch_ball_xy,
face_color=sv.Color.WHITE,
edge_color=sv.Color.WHITE,
radius=8,
thickness=1,
pitch=annotated_frame)
annotated_frame = draw_points_on_pitch(
config=CONFIG,
xy=pitch_players_xy[players_detections.class_id == 0],
face_color=sv.Color.from_hex('00BFFF'),
edge_color=sv.Color.WHITE,
radius=16,
thickness=1,
pitch=annotated_frame)
annotated_frame = draw_points_on_pitch(
config=CONFIG,
xy=pitch_players_xy[players_detections.class_id == 1],
face_color=sv.Color.from_hex('FF1493'),
edge_color=sv.Color.WHITE,
radius=16,
thickness=1,
pitch=annotated_frame)

sv.plot_image(annotated_frame)
