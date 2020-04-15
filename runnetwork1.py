#运行network1
import mnist_load
import network1
training_data,validation_data,test_data = mnist_load.load_data_wrapper()
net = network1.Network([784,30,10])
net.SGD(training_data, 10, 10, 3.0,test_data = test_data)