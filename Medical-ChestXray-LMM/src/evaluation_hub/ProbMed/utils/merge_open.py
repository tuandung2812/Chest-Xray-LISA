#         messages_for_cls = [
#             {
#                 "role": "system", 
#                 "content": f"""You are an expert chest xray radiologist. 
#                     You are not given image. 
#                     The only information you have is the <abnormalities list>.
#                     You need to classify a <abnormality> in the <abnormalities list> with one of the: {", ".join(VINDR_LABEL)}
#                     Keep exactly the name of the list.
#                     Remove <abnormality> if this is not in the list.
#                     Answer in this format: abnormality 1, abnormality 2, abnormality 3.
#                     Remember 'ground-glass opacities' is 'lung opacification'.
#                     Remember 'ground-glass opacities' is 'lung opacification'.
#                 """
#             },
#             {
#                 "role": "user", 
#                 "content": """This is the <abnormalities list> '{question}'
#                 """
#             }
#         ]

# messages_for_cls_new = messages_for_cls.copy()
# messages_for_cls_new[1]["content"] = messages_for_cls_new[1]["content"].replace("{question}", prediction_extracted)
# prediction_cls = evaluator.inference(
# messages=messages_for_cls_new
# )

#                 from thefuzz import fuzz
#                 prediction_cls = prediction_extracted
#                 prediction_cls_post = []
#                 for disease in prediction_cls.split(", "):
#                     if disease in ["ground-glass opacities", "ground-glass opacification", "ground-glass opacity", "ground glass opacities"]:
#                         disease = "lung opacification"
#                     elif disease in ["patchy infiltrates"]:
#                         disease = "infiltration"
#                     highest_map = 0
#                     best_match = None
#                     for target_disease in VINDR_LABEL:
#                         map_ratio = fuzz.partial_ratio(disease, target_disease)
#                         if map_ratio > highest_map:
#                             highest_map = map_ratio
#                             best_match = target_disease
#                     if highest_map > 95:
#                         prediction_cls_post.append(best_match)
#                 prediction_cls_post = ", ".join(prediction_cls_post)
#                 data_predict.loc[idx, 'prediction_cls'] = prediction_cls_post
#                 pred_onehot = [1 if x in prediction_cls else 0 for x in VINDR_LABEL]