
## Tensorflow Playground ##

----------

###Summary

I started this project as a simple sandbox to experiment and apply Machine Learning concepts with bare-metal TensorFlow. 
Over time, I plan to implement more examples and techniques to explore various topics in ML.

Please feel free to use, study, modify them and point out any improvement/error you may find.

----------

###Results


Currently the following examples are implemented:
 
***MNIST*** 
[Yann LeCun MNIST homepage - Download Dataset](http://yann.lecun.com/exdb/mnist/)
[Tensorflow Tutorials](https://www.tensorflow.org/get_started/mnist/pros)

| Method      | Accuracy | Executable       |
| ------      | -------- | ----------       |
| Linear      | ~ 92     | mnist_logit.py   |
| Neural Net  | ~ 98.5   | mnist_nn.py      |
| CNN LeNet-5 | ~ 99.2   | mnist_convnet.py |	


----------

###Usage
First of all, the proper datasets data shall be downloaded from the appropriate source, and the files location shall be configured into each executable file.
As an example, using MNIST will require downloading it in the **mnist** folder and configuring by

    dset_loader = LoaderMnist("./mnist/train-images-idx3-ubyte",
                              "./mnist/train-labels-idx1-ubyte",
                              "./mnist/t10k-images-idx3-ubyte",
                              "./mnist/t10k-labels-idx1-ubyte",
                              validation_percent,
                              limit_images)

To run an example, just launch

    python <executable>

where ***executable*** is one of the provided files in the table of the Results section.
Each executable will try to automatically use all GPUs, or fallback to the cpus if no gpu is available.

**At the end of the run, the filters/neurons are plotted on screen for debug purposes.**
To disable this behavior, comment the proper *Display* functions at the end of each executable.
 
All results were tested under Arch Linux

----------

###Hyperparameters

If you want to tweak the dataset split, networks structure and hyperparameters, you can find them at the beginning of each executable in each section marked with a comment ending like  `==> USER`
The code should be already self-explanatory.

----------


###License

See the LICENSE file for license rights and limitations (MIT).

