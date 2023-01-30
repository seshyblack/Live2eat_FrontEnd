import time
import statistics
import numpy as np
import pandas as pd
import cv2
import numpy as np
import pickle
import glob
import os
import streamlit as st

from typing import Optional, Sequence
from datetime import timedelta
from google.oauth2 import service_account
from sklearn.cluster import KMeans

from google.cloud import videointelligence as vi
from google.cloud import storage


def track_objects(
    video_uri: str,
    credentials,
    segments: Optional[Sequence[vi.VideoSegment]] = None
) -> vi.VideoAnnotationResults:

    video_client = vi.VideoIntelligenceServiceClient(credentials=credentials)
    features = [vi.Feature.OBJECT_TRACKING]
    context = vi.VideoContext(segments=segments)
    request = vi.AnnotateVideoRequest(
        input_uri=video_uri,
        features=features,
        video_context=context,
    )
    print(f'Processing video "{video_uri}"...')

    operation = video_client.annotate_video(request)
    return operation.result().annotation_results[0]


def print_object_frames(results: vi.VideoAnnotationResults,
                        entity_id: str,
                        min_confidence: float = 0.7):

    def keep_annotation(annotation: vi.ObjectTrackingAnnotation) -> bool:
        return all([
            annotation.entity.entity_id == entity_id,
            min_confidence <= annotation.confidence,
        ])

    annotations = results.object_annotations
    annotations = [a for a in annotations if keep_annotation(a)]
    object_frames = []

    for annotation in annotations:

        for frame in annotation.frames:
            t = frame.time_offset.total_seconds()
            object_frames.append(t)

    return object_frames
