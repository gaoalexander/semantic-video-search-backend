import os
import sys
import json
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from embeddings.textmatching import get_embedding
from embeddings.text_feature import text_feature


class QueryExecutor:
	def __init__(self, search_query):
		self.embedder = text_feature(mode = "single")
		self.class_vector_path = "/Users/alexandergao/Desktop/pytorch_hackathon/python/embeddings/class_vector.pkl"
		self.obj_detect_path = "/Users/alexandergao/Desktop/pytorch_hackathon/video_data/post_detection/group_results.p"
		self.results_output_path = "/Users/alexandergao/Desktop/pytorch_hackathon/to_frontend/detected_frames.txt"
		self.detected_class_dict = {}
		self.threshold = 0.75
		self.query_detected_frames = {}
		self.class_vectors = self.load_pickle(self.class_vector_path)
		self.obj_detect_data = self.load_pickle(self.obj_detect_path)
		self.cleaned_obj_dict = self.format_raw_obj_detect_data(self.obj_detect_data)
		self.search_query = search_query

	def load_pickle(self, input_path):
		return pickle.load(open(input_path, "rb"))

	def format_raw_obj_detect_data(self, obj_detect_data):
		'''
		For a given object detected in a single video, only return frame indexes that are at least 10 secons apart
		'''
		for frame in obj_detect_data:
			frame_id = frame["video"].split("/")[-1].replace(".mp4", "") + "_" + str(frame["frame"])
			classes = set(frame['results'][1])
			for _class in classes:
				if _class in self.detected_class_dict:
					self.detected_class_dict[_class].append(frame_id)
				else:
					self.detected_class_dict[_class] =[frame_id]

		'''
		filter out frame_id's that are within ten seconds of each other
		'''
		chunk_length = 240
		cleaned_obj_dict = {}
		for _class in self.detected_class_dict:
			current_index = -250
			for frame_id in self.detected_class_dict[_class]:
				video_id = frame_id.split("_")[0]
				frame_index = frame_id.split("_")[-1]
				if abs(int(frame_index) - int(current_index)) > chunk_length:
					if _class in cleaned_obj_dict:
						cleaned_obj_dict[_class].append(frame_id)
					else:
						cleaned_obj_dict[_class] = [frame_id]
					current_index = frame_index
		return cleaned_obj_dict

	def write_output(self, query_detected_frames, results_output_path):
		with open(results_output_path, 'w') as outfile:
			json.dump(query_detected_frames, outfile)

	def get_frames(self, search_query):
		query_embedding = get_embedding(search_query, self.embedder)

		cos_sim_max = 0
		for class_name in self.class_vectors:
			known_class_embedding = self.class_vectors[class_name]
			cos_sim = cosine_similarity(query_embedding, known_class_embedding)[0][0]
			if cos_sim > cos_sim_max:
				cos_sim_max = cos_sim
				most_likely_class = class_name

		if most_likely_class in self.cleaned_obj_dict:
			self.query_detected_frames['class'] = most_likely_class
			self.query_detected_frames['detected_frames'] = {}
			for frame in self.cleaned_obj_dict[most_likely_class]:
				frame = frame.split("_")
				video_id = frame[0] + ".mp4"
				frame_index = frame[-1]
				if video_id in self.query_detected_frames['detected_frames']:
					self.query_detected_frames['detected_frames'][video_id].append(int(frame_index)//24)
				else:
					self.query_detected_frames['detected_frames'][video_id] = [int(frame_index)//24]
			return self.query_detected_frames
		else:
			print("Query Not Found.")
			return self.query_detected_frames

	def execute_search_query(self):
	
		query_detected_frames = self.get_frames(self.search_query)
		self.write_output(self.query_detected_frames, self.results_output_path)

# ----------------------------------------------------------------------------------------------

if __name__ == "__main__":
	query_executor = QueryExecutor(sys.argv[1])
	query_executor.execute_search_query()



	# if cos_sim > threshold:
	# 	frame_candidates.append(video_frame_data[i])

