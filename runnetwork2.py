import mnist_load
import network2
training_data,validation_data,test_data = mnist_load.load_data_wrapper()
net = network2.Network([784,30,10],cost = network2.CrossEntropyCost)
net.SGD(training_data,10,10,0.5,evaluation_data = test_data,
        monitor_evaluation_accuracy=True)
