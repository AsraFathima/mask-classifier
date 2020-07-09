from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry
from msrest.authentication import ApiKeyCredentials
import time


 
ENDPOINT = "https://face-mask-prediction.cognitiveservices.azure.com/"

# Replace with a valid key
training_key = "eb9e574c5a3248b3bf86dc50e52029ac"
prediction_key = "ffdd7ab5e60a4d71aa930e2b813b5b29"
prediction_resource_id = "/subscriptions/5ba799a2-4264-4a21-bd20-1be80766d2f0/resourceGroups/AzurePy/providers/Microsoft.CognitiveServices/accounts/facemaskprediction-Prediction"

publish_iteration_name = "classifyModel"

credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})

trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

#Create a new project
print ("Creating project...")
#need not create always
project = trainer.create_project("test") 

withMask_tag = trainer.create_tag(project.id, "With Mask")
withoutMask_tag = trainer.create_tag(project.id, "Without Mask")

base_image_url = "./images/"

print("Adding images...")

image_list = []

for image_num in range(1, 7):
    file_name = "withmask_{}.jpg".format(image_num)
    with open(base_image_url + "withmask/" + file_name, "rb") as image_contents:
        image_list.append(ImageFileCreateEntry(name=file_name, contents=image_contents.read(), tag_ids=[withMask_tag.id]))

for image_num in range(1, 7):
    file_name = "withoutmask_{}.jpg".format(image_num)
    with open(base_image_url + "withoutmask/" + file_name, "rb") as image_contents:
        image_list.append(ImageFileCreateEntry(name=file_name, contents=image_contents.read(), tag_ids=[withoutMask_tag.id]))

upload_result = trainer.create_images_from_files(project.id, ImageFileCreateBatch(images=image_list))
if not upload_result.is_batch_successful:
    print("Image batch upload failed.")
    for image in upload_result.images:
        print("Image status: ", image.status)
    exit(-1)

print ("Training...")
iteration = trainer.train_project(project.id)
while (iteration.status != "Completed"):
    iteration = trainer.get_iteration(project.id, iteration.id)
    print ("Training status: " + iteration.status)
    time.sleep(1)

# The iteration is now trained. Publish it to the project endpoint
trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)
print ("Done!")

# Now there is a trained endpoint that can be used to make a prediction
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)
project_id=project.id

with open(base_image_url + "withoutmask/withoutmask_6.jpg", "rb") as image_contents:
    results = predictor.classify_image(
        project_id, publish_iteration_name, image_contents.read())

    # Display the results.
    for prediction in results.predictions:
        print("\t" + prediction.tag_name +
              ": {0:.2f}%".format(prediction.probability * 100))
print(project.id)