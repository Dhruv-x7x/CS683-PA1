#include "nn.h"

Matrix Layer::forward(const Matrix &input) {
	lastInput = input; 

	// Compute the weighted sum
	// sum = weights * input + biases
	Matrix weightedSum = MatrixOperation::MatMul(weights, input, optLevel);
	// Add biases
	float b = 0.0f;
	for(size_t i = 0; i < weightedSum.getRows(); ++i) {
		b = biases(i, 0);	
		for (size_t j = 0; j < weightedSum.getCols(); ++j) {
			weightedSum(i, j) += b;
		}
	}

	// Apply activation function
	
	lastOutput = ActivationFunction::apply(weightedSum, activationFunc);
	return lastOutput;
}

Matrix Layer::backward(const Matrix &outputGradient, double learningRate) {
	// Compute activation gradient
	Matrix activationGrad (lastOutput.getRows(), lastOutput.getCols());
	for (size_t i = 0; i < lastOutput.getRows(); ++i) {
		for (size_t j = 0; j < lastOutput.getCols(); ++j) {
			activationGrad(i, j) = outputGradient(i, j) * activationDerivative(lastOutput(i, j));
		}
	}

	// compute weight gradients
	Matrix inputTranspose = MatrixOperation::Transpose(lastInput);
	Matrix weightGradients = MatrixOperation::MatMul(activationGrad, inputTranspose, optLevel);

	// Compute input gradients
	Matrix weightsTransposed = MatrixOperation::Transpose(weights);
	Matrix inputGradients = MatrixOperation::MatMul(weightsTransposed, activationGrad, optLevel);

	// Update weights and biases
	// weights -= weightGradients * learningRate;
	// biases -= activationGrad * learningRate;
	for(size_t i = 0; i < weights.getRows(); ++i) {
		for (size_t j = 0; j < weights.getCols(); ++j) {
			weights(i, j) -= weightGradients(i, j) * learningRate;
		}
	}

	// Update biases
	for(size_t i =0; i < biases.getRows(); ++i) {
		double biasGrad = 0.0;
		for (size_t j = 0; j < biases.getCols(); ++j) {
			biasGrad += activationGrad(i, j);
		}
		biases(i, 0) -= biasGrad * learningRate/ activationGrad.getCols();
	}

	return inputGradients;
}


Matrix NeuralNetwork::predict(const Matrix &input) {
	Matrix current = input ;
	for(auto &layer : layers){
		current = layer.forward(current);
	}
	return current;
}

void NeuralNetwork::train(const Matrix &input, const Matrix &target, double learningRate){
	//Forward pass
	Matrix prediction = predict(input);

	// Compute loss gradient
	Matrix lossGradient(prediction.getRows(), prediction.getCols());
	for (size_t i = 0; i < prediction.getRows(); ++i) {
		for (size_t j = 0; j < prediction.getCols(); ++j) {
			lossGradient(i, j) = 2.0 * (prediction(i,j) - target(i, j));
		}
	}

	// Backward pass
	Matrix currentGradient = lossGradient;
	for(int i = layers.size() - 1; i>= 0; --i) {
		currentGradient = layers[i].backward(currentGradient, learningRate);
	}
}

double NeuralNetwork::computeLoss(const Matrix &input, const Matrix &target){
	Matrix prediction = predict(input);
	double loss = 0.0;
	double count = 0.0;

	for(size_t i = 0; i< prediction.getRows(); ++i){
		for (size_t j = 0 ; j < prediction.getCols(); ++j) {
			double diff = prediction(i, j) - target(i, j);
			loss += diff * diff;
			count++;
		}
	}

	cout<<"Total loss: " << loss << " over " << count << " elements." << endl;
	return loss/count; 
}

void NeuralNetwork::setOptimization(MatrixOptimization opt) {
	optLevel = opt;
	for (auto &layer: layers){
		layer.setOptimization(opt);
	}
}



	

