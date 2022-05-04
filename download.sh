# Downloads the zip file that contains the data
curl https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz --output cifar-10-python.tar.gz
# Unzips the zip file - you will get a directory named "cifar-10-batches-py" containing the data
tar -xzvf cifar-10-python.tar.gz
# Removes the zip file, as we will no longer use it
rm cifar-10-python.tar.gz
# Prints a success message
echo "CIFAR-10 data successfully downloaded"