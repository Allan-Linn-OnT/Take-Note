# Take-Note
A machine learning model designed to turn you into a professional musician!

This branch contains the code required to train, evaluate, and generate samples from the original Coconet GitHub. 

The coconet_train.py file performs all the training code, the other files contain external parts such as the model definition and sampling functions.

Some of the files from the original Coconet GitHub were reused as they deal with numpy not tensorflow.

The final models are exported to [ONNX](https://github.com/onnx/onnx) format such that they can be loaded into the sampling / inference code from the Coconet GitHub.

# Instructions
<ol>
<li>Clone the original Coconet [GitHub](https://github.com/magenta/magenta/tree/main/magenta/models/coconet)</li>

<li>Clone this repo into the root directory of the previous GitHub.</li>

<li>Use the sample_bazel.sh and train_bazel.sh scripts to call the coconet_train.py function from this repo.</li>
</ol>