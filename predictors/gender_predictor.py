import pandas as pd

def simple_gender_predictor(uid, image_data):
    uid_data = image_data[image_data['userId']==uid]
    if len(uid_data) == 1:
        mustache = image_data[image_data['userId']==uid]['facialHair_mustache'].iloc[0]
        beard = image_data[image_data['userId']==uid]['facialHair_beard'].iloc[0]
        sideburns = image_data[image_data['userId']==uid]['facialHair_sideburns'].iloc[0]

        if beard + mustache + sideburns > 0:
            # Predict male
            return 0

        else:
            # Predict female
            return 1
    
    else:
        # Predict baseline
        return 2