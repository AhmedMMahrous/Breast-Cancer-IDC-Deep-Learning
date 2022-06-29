from flask import Flask, render_template, request # Import Flask
from keras.models import load_model	  # Import the model we saved
from keras.preprocessing import image  # Importing the Keras libraries and packages
import numpy as np # Import numpy

app = Flask(__name__)  # Initialize the Flask application

dic = {0 : 'She doesnot have cancer', 1 : 'She has cancer'}	  # Dictionary to map the model output to the actual label

model = load_model('model.h5') # Load the model from the saved file

model.make_predict_function()   # This is necessary for model to be used in a flask app

def predict_label(img_path): # Function to predict the label of the image
	i = image.load_img(img_path, target_size=(25,25)) # Load the image
	i = image.img_to_array(i)/255.0 # Convert the image to an array
	i = i.reshape(1, 25,25,3)	# Reshape the array to be compatible with the model
	p = np.argmax(model.predict(i), axis=-1) 	# Predict the label of the image
	return dic[p[0]]			# Return the label


# routes
@app.route("/", methods=['GET', 'POST'])		# The function called when the root path is called
def main():							# The function called when the root path is called
	return render_template('index.html')	# Render the template


@app.route("/submit", methods = ['GET', 'POST'])	# The function called when the submit path is called	
def get_output():					# The function called when the submit path is called
	if request.method == 'POST':		# If the request method is POST
		img = request.files['my_image']	# Get the image from the form

		img_path = "static/" + img.filename		# Create the path to the image
		img.save(img_path)		# Save the image to the path

		p = predict_label(img_path)	# Predict the label of the image

	return render_template('index.html', prediction = p, img_path = img_path)	# Render the template with the prediction and the image path


if __name__ =='__main__':	# The function called when the script is run
	#app.debug = True
	app.run(debug = True)	# Run the app in debug mode