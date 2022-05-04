# this downloads the zip file that contains the data
curl https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz --output cifar-10-python.tar.gz
# this unzips the zip file - you will get a directory named "data" containing the data
tar -xzvf cifar-10-python.tar.gz
# this cleans up the zip file, as we will no longer use it
rm cifar-10-python.tar.gz

echo "CIFAR-10 data successfully downloaded"