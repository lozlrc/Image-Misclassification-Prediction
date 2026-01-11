# Image-Misclassification-Prediction
It builds a CIFAR-10 image classifier, then train a second model that predicts when the classifier will be wrong. Use this “failure risk” to abstain on uncertain images and improve reliability with clear evaluation plots and a Streamlit demo.

No Kaggle or external credentials are used.

Your filenames may differ. Update the commands below to match your scripts if needed.

## How to run:

install the following packages:

	- pip install torch torchvision torchaudio
	- pip install numpy pandas scikit-learn matplotlib tqdm
	- pip install streamlit pillow


Step 1: Train the base image classifier
python3 -m src.train_base --device mps --epochs 6

Step 2: Generate the meta-dataset from base model outputs
python3 -m src.make_meta_dataset --device mps

Step 3: Train the failure predictor (meta-model)
python3 -m src.train_meta

Step 4: Evaluate and generate plots
python3 -m src.eval_base
python3 -m src.eval_selective

Step 5: Launch the Streamlit demo
python3 -m streamlit run app/streamlit_app.py

Streamlit demo:

	- a CIFAR-10 test image by index
	- top-3 predictions
	- predicted failure risk p_fail
	- abstain / predict decision based on threshold
	- abstention precision and number abstained on the test set
