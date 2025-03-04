import { Matrix } from 'ml-matrix';
import { ModelType, TrainingData, ModelResults } from '../types';

/**
 * Train a linear regression model
 */
export function trainLinearModel(trainingData: TrainingData, hyperparameters?: any): ModelResults {
  const { X_train, y_train, X_test, y_test, featureNames, isClassification, classes } = trainingData;
  
  // Get hyperparameters
  const regularization = hyperparameters?.regularization || 0.01;
  const learningRate = hyperparameters?.learningRate || 0.01;
  const iterations = hyperparameters?.iterations || 100;
  
  if (isClassification) {
    // For classification, use logistic regression
    return trainLogisticRegression(
      X_train, y_train, X_test, y_test, featureNames, 
      classes || [], regularization, learningRate, iterations
    );
  } else {
    // For regression, use linear regression
    return trainLinearRegression(
      X_train, y_train, X_test, y_test, featureNames, 
      regularization, learningRate, iterations
    );
  }
}

/**
 * Train a linear regression model for regression tasks
 */
function trainLinearRegression(
  X_train: number[][], 
  y_train: number[][], 
  X_test: number[][], 
  y_test: number[][],
  featureNames: string[],
  regularization: number = 0.01,
  learningRate: number = 0.01,
  iterations: number = 100
): ModelResults {
  try {
    const n = X_train[0].length; // Number of features
    const m = X_train.length;    // Number of training examples
    const numOutputs = y_train[0].length; // Number of output variables
    
    // Initialize weights and bias
    let weights = Array(numOutputs).fill(0).map(() => Array(n).fill(0));
    let biases = Array(numOutputs).fill(0);
    
    // Normalize features
    const { X_train_norm, X_test_norm, means, stds } = normalizeFeatures(X_train, X_test);
    
    // Gradient descent
    for (let iter = 0; iter < iterations; iter++) {
      // For each output variable
      for (let k = 0; k < numOutputs; k++) {
        // Compute predictions
        const predictions = X_train_norm.map((x, i) => {
          let pred = biases[k];
          for (let j = 0; j < n; j++) {
            pred += weights[k][j] * x[j];
          }
          return pred;
        });
        
        // Compute gradients
        const dw = Array(n).fill(0);
        let db = 0;
        
        for (let i = 0; i < m; i++) {
          const error = predictions[i] - y_train[i][k];
          
          // Gradient for bias
          db += error;
          
          // Gradient for weights
          for (let j = 0; j < n; j++) {
            dw[j] += error * X_train_norm[i][j];
          }
        }
        
        // Update parameters with regularization
        biases[k] -= learningRate * (db / m);
        
        for (let j = 0; j < n; j++) {
          weights[k][j] -= learningRate * ((dw[j] / m) + regularization * weights[k][j]);
        }
      }
    }
    
    // Make predictions on test set
    const predictions = X_test_norm.map(x => {
      const pred = Array(numOutputs).fill(0);
      
      for (let k = 0; k < numOutputs; k++) {
        pred[k] = biases[k];
        for (let j = 0; j < n; j++) {
          pred[k] += weights[k][j] * x[j];
        }
      }
      
      return pred;
    });
    
    // Calculate metrics
    const mse = calculateMSE(predictions, y_test);
    const r2 = calculateR2(predictions, y_test);
    
    // Calculate feature importance
    const featureImportance = calculateLinearFeatureImportance(weights, featureNames);
    
    return {
      modelType: 'linear',
      modelName: `多元線性回歸 (L2=${regularization})`,
      mse,
      r2,
      featureImportance,
      predictions,
      actual: y_test
    };
  } catch (error) {
    console.error('Error in linear model training:', error);
    throw error;
  }
}

/**
 * Train a logistic regression model for classification tasks
 */
function trainLogisticRegression(
  X_train: number[][], 
  y_train: number[][], 
  X_test: number[][], 
  y_test: number[][],
  featureNames: string[],
  classes: (string | number)[],
  regularization: number = 0.01,
  learningRate: number = 0.01,
  iterations: number = 100
): ModelResults {
  try {
    const n = X_train[0].length; // Number of features
    const m = X_train.length;    // Number of training examples
    const numClasses = classes.length || Math.max(...y_train.map(y => y[0])) + 1;
    
    // For binary classification
    const isBinary = numClasses <= 2;
    const actualNumClasses = isBinary ? 1 : numClasses;
    
    // Initialize weights and bias
    let weights = Array(actualNumClasses).fill(0).map(() => Array(n).fill(0));
    let biases = Array(actualNumClasses).fill(0);
    
    // Normalize features
    const { X_train_norm, X_test_norm, means, stds } = normalizeFeatures(X_train, X_test);
    
    // Sigmoid function
    const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
    
    // Softmax function for multi-class
    const softmax = (z: number[]) => {
      const expValues = z.map(val => Math.exp(val));
      const sumExp = expValues.reduce((a, b) => a + b, 0);
      return expValues.map(val => val / sumExp);
    };
    
    // Gradient descent
    for (let iter = 0; iter < iterations; iter++) {
      if (isBinary) {
        // Binary classification
        // Compute predictions
        const predictions = X_train_norm.map((x, i) => {
          let z = biases[0];
          for (let j = 0; j < n; j++) {
            z += weights[0][j] * x[j];
          }
          return sigmoid(z);
        });
        
        // Compute gradients
        const dw = Array(n).fill(0);
        let db = 0;
        
        for (let i = 0; i < m; i++) {
          const error = predictions[i] - (y_train[i][0] === 1 ? 1 : 0);
          
          // Gradient for bias
          db += error;
          
          // Gradient for weights
          for (let j = 0; j < n; j++) {
            dw[j] += error * X_train_norm[i][j];
          }
        }
        
        // Update parameters with regularization
        biases[0] -= learningRate * (db / m);
        
        for (let j = 0; j < n; j++) {
          weights[0][j] -= learningRate * ((dw[j] / m) + regularization * weights[0][j]);
        }
      } else {
        // Multi-class classification
        // For each class
        for (let k = 0; k < numClasses; k++) {
          // Compute all class scores for each example
          const allScores = X_train_norm.map((x, i) => {
            return Array(numClasses).fill(0).map((_, c) => {
              let score = biases[c];
              for (let j = 0; j < n; j++) {
                score += weights[c][j] * x[j];
              }
              return score;
            });
          });
          
          // Apply softmax to get probabilities
          const probabilities = allScores.map(scores => softmax(scores));
          
          // Compute gradients for current class
          const dw = Array(n).fill(0);
          let db = 0;
          
          for (let i = 0; i < m; i++) {
            // One-hot encoding: 1 for the true class, 0 for others
            const trueClass = y_train[i][0];
            const error = probabilities[i][k] - (trueClass === k ? 1 : 0);
            
            // Gradient for bias
            db += error;
            
            // Gradient for weights
            for (let j = 0; j < n; j++) {
              dw[j] += error * X_train_norm[i][j];
            }
          }
          
          // Update parameters with regularization
          biases[k] -= learningRate * (db / m);
          
          for (let j = 0; j < n; j++) {
            weights[k][j] -= learningRate * ((dw[j] / m) + regularization * weights[k][j]);
          }
        }
      }
    }
    
    // Make predictions on test set
    let predictions: number[][];
    
    if (isBinary) {
      // Binary classification
      predictions = X_test_norm.map(x => {
        let z = biases[0];
        for (let j = 0; j < n; j++) {
          z += weights[0][j] * x[j];
        }
        const probability = sigmoid(z);
        return [probability >= 0.5 ? 1 : 0];
      });
    } else {
      // Multi-class classification
      predictions = X_test_norm.map(x => {
        const scores = Array(numClasses).fill(0).map((_, k) => {
          let score = biases[k];
          for (let j = 0; j < n; j++) {
            score += weights[k][j] * x[j];
          }
          return score;
        });
        
        const probs = softmax(scores);
        const predictedClass = probs.indexOf(Math.max(...probs));
        return [predictedClass];
      });
    }
    
    // Calculate metrics
    const accuracy = calculateAccuracy(predictions, y_test);
    const { precision, recall, f1Score } = calculateClassificationMetrics(
      predictions, y_test, numClasses
    );
    const confusionMatrix = calculateConfusionMatrix(predictions, y_test, numClasses);
    
    // Calculate feature importance
    const featureImportance = calculateLinearFeatureImportance(weights, featureNames);
    
    return {
      modelType: 'linear',
      modelName: `邏輯迴歸 (L2=${regularization})`,
      mse: 0, // Not relevant for classification
      r2: 0,  // Not relevant for classification
      accuracy,
      precision,
      recall,
      f1Score,
      confusionMatrix,
      featureImportance,
      predictions,
      actual: y_test,
      isClassification: true,
      classes
    };
  } catch (error) {
    console.error('Error in logistic regression training:', error);
    throw error;
  }
}

/**
 * Train a decision tree model
 */
export function trainDecisionTreeModel(trainingData: TrainingData, hyperparameters?: any): ModelResults {
  const { X_train, y_train, X_test, y_test, featureNames, isClassification, classes } = trainingData;
  
  // Get hyperparameters
  const maxDepth = hyperparameters?.maxDepth || 5;
  const minSamplesSplit = hyperparameters?.minSamplesSplit || 2;
  
  if (isClassification) {
    return trainDecisionTreeClassifier(
      X_train, y_train, X_test, y_test, featureNames,
      classes || [], maxDepth, minSamplesSplit
    );
  } else {
    return trainDecisionTreeRegressor(
      X_train, y_train, X_test, y_test, featureNames,
      maxDepth, minSamplesSplit
    );
  }
}

/**
 * Train a decision tree for regression
 */
function trainDecisionTreeRegressor(
  X_train: number[][], 
  y_train: number[][], 
  X_test: number[][], 
  y_test: number[][],
  featureNames: string[],
  maxDepth: number = 5,
  minSamplesSplit: number = 2
): ModelResults {
  try {
    // For simplicity, we'll use the first target variable if there are multiple
    const y_train_first = y_train.map(row => row[0]);
    
    // Build the tree
    const tree = buildRegressionTree(
      X_train, 
      y_train_first, 
      0, 
      maxDepth, 
      minSamplesSplit,
      featureNames
    );
    
    // Make predictions
    const predictions = X_test.map(sample => {
      const prediction = predictFromTree(sample, tree);
      return [prediction];
    });
    
    // Calculate metrics
    const mse = calculateMSE(predictions, y_test);
    const r2 = calculateR2(predictions, y_test);
    
    // Get feature importance
    const featureImportance = calculateTreeFeatureImportance(tree, featureNames);
    
    return {
      modelType: 'decisionTree',
      modelName: `決策樹回歸 (深度=${maxDepth})`,
      mse,
      r2,
      featureImportance,
      predictions,
      actual: y_test
    };
  } catch (error) {
    console.error('Error in decision tree regression:', error);
    throw error;
  }
}

/**
 * Train a decision tree for classification
 */
function trainDecisionTreeClassifier(
  X_train: number[][], 
  y_train: number[][], 
  X_test: number[][], 
  y_test: number[][],
  featureNames: string[],
  classes: (string | number)[],
  maxDepth: number = 5,
  minSamplesSplit: number = 2
): ModelResults {
  try {
    // For classification, we use the first column of y
    const y_train_first = y_train.map(row => row[0]);
    
    // Build the tree
    const tree = buildClassificationTree(
      X_train, 
      y_train_first, 
      0, 
      maxDepth, 
      minSamplesSplit,
      featureNames,
      classes.length
    );
    
    // Make predictions
    const predictions = X_test.map(sample => {
      const prediction = predictFromTree(sample, tree);
      return [prediction];
    });
    
    // Calculate metrics
    const accuracy = calculateAccuracy(predictions, y_test);
    const { precision, recall, f1Score } = calculateClassificationMetrics(
      predictions, y_test, classes.length
    );
    const confusionMatrix = calculateConfusionMatrix(predictions, y_test, classes.length);
    
    // Get feature importance
    const featureImportance = calculateTreeFeatureImportance(tree, featureNames);
    
    return {
      modelType: 'decisionTree',
      modelName: `決策樹分類 (深度=${maxDepth})`,
      mse: 0, // Not relevant for classification
      r2: 0,  // Not relevant for classification
      accuracy,
      precision,
      recall,
      f1Score,
      confusionMatrix,
      featureImportance,
      predictions,
      actual: y_test,
      isClassification: true,
      classes
    };
  } catch (error) {
    console.error('Error in decision tree classification:', error);
    throw error;
  }
}

/**
 * Build a decision tree for regression
 */
function buildRegressionTree(
  X: number[][], 
  y: number[], 
  depth: number, 
  maxDepth: number,
  minSamplesSplit: number,
  featureNames: string[]
): any {
  const m = X.length;
  const n = X[0].length;
  
  // If we've reached max depth or have too few samples, create a leaf node
  if (depth >= maxDepth || m < minSamplesSplit) {
    // Return the mean value as prediction
    const mean = y.reduce((sum, val) => sum + val, 0) / m;
    return {
      isLeaf: true,
      value: mean,
      samples: m
    };
  }
  
  // Find the best split
  let bestGain = -Infinity;
  let bestFeature = 0;
  let bestThreshold = 0;
  let bestLeftX: number[][] = [];
  let bestLeftY: number[] = [];
  let bestRightX: number[][] = [];
  let bestRightY: number[] = [];
  
  // Calculate variance of current node
  const mean = y.reduce((sum, val) => sum + val, 0) / m;
  const variance = y.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / m;
  
  // Try each feature
  for (let feature = 0; feature < n; feature++) {
    // Get unique values for this feature
    const values = [...new Set(X.map(row => row[feature]))].sort((a, b) => a - b);
    
    // Try each value as a threshold
    for (let i = 0; i < values.length - 1; i++) {
      const threshold = (values[i] + values[i + 1]) / 2;
      
      // Split the data
      const leftIndices = [];
      const rightIndices = [];
      
      for (let j = 0; j < m; j++) {
        if (X[j][feature] <= threshold) {
          leftIndices.push(j);
        } else {
          rightIndices.push(j);
        }
      }
      
      // Skip if either split is empty
      if (leftIndices.length === 0 || rightIndices.length === 0) continue;
      
      // Calculate information gain
      const leftY = leftIndices.map(idx => y[idx]);
      const rightY = rightIndices.map(idx => y[idx]);
      
      const leftMean = leftY.reduce((sum, val) => sum + val, 0) / leftY.length;
      const rightMean = rightY.reduce((sum, val) => sum + val, 0) / rightY.length;
      
      const leftVariance = leftY.reduce((sum, val) => sum + Math.pow(val - leftMean, 2), 0) / leftY.length;
      const rightVariance = rightY.reduce((sum, val) => sum + Math.pow(val - rightMean, 2), 0) / rightY.length;
      
      // Weighted variance reduction
      const leftWeight = leftY.length / m;
      const rightWeight = rightY.length / m;
      const weightedVariance = leftWeight * leftVariance + rightWeight * rightVariance;
      
      const gain = variance - weightedVariance;
      
      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = feature;
        bestThreshold = threshold;
        bestLeftX = leftIndices.map(idx => X[idx]);
        bestLeftY = leftY;
        bestRightX = rightIndices.map(idx => X[idx]);
        bestRightY = rightY;
      }
    }
  }
  
  // If no good split found, create a leaf node
  if (bestGain <= 0) {
    return {
      isLeaf: true,
      value: mean,
      samples: m
    };
  }
  
  // Create a decision node
  const leftChild = buildRegressionTree(
    bestLeftX, bestLeftY, depth + 1, maxDepth, minSamplesSplit, featureNames
  );
  
  const rightChild = buildRegressionTree(
    bestRightX, bestRightY, depth + 1, maxDepth, minSamplesSplit, featureNames
  );
  
  return {
    isLeaf: false,
    feature: bestFeature,
    featureName: featureNames[bestFeature],
    threshold: bestThreshold,
    gain: bestGain,
    left: leftChild,
    right: rightChild,
    samples: m,
    depth: depth
  };
}

/**
 * Build a decision tree for classification
 */
function buildClassificationTree(
  X: number[][], 
  y: number[], 
  depth: number, 
  maxDepth: number,
  minSamplesSplit: number,
  featureNames: string[],
  numClasses: number
): any {
  const m = X.length;
  const n = X[0].length;
  
  // If we've reached max depth or have too few samples, create a leaf node
  if (depth >= maxDepth || m < minSamplesSplit) {
    // Return the most frequent class
    const classCounts = Array(numClasses).fill(0);
    for (const cls of y) {
      classCounts[cls]++;
    }
    const predictedClass = classCounts.indexOf(Math.max(...classCounts));
    
    return {
      isLeaf: true,
      value: predictedClass,
      classCounts,
      samples: m
    };
  }
  
  // Calculate current entropy/gini impurity
  const classCounts = Array(numClasses).fill(0);
  for (const cls of y) {
    classCounts[cls]++;
  }
  
  // Use Gini impurity
  const gini = 1 - classCounts.reduce((sum, count) => sum + Math.pow(count / m, 2), 0);
  
  // Find the best split
  let bestGain = -Infinity;
  let bestFeature = 0;
  let bestThreshold = 0;
  let bestLeftX: number[][] = [];
  let bestLeftY: number[] = [];
  let bestRightX: number[][] = [];
  let bestRightY: number[] = [];
  
  // Try each feature
  for (let feature = 0; feature < n; feature++) {
    // Get unique values for this feature
    const values = [...new Set(X.map(row => row[feature]))].sort((a, b) => a - b);
    
    // Try each value as a threshold
    for (let i = 0; i < values.length - 1; i++) {
      const threshold = (values[i] + values[i + 1]) / 2;
      
      // Split the data
      const leftIndices = [];
      const rightIndices = [];
      
      for (let j = 0; j < m; j++) {
        if (X[j][feature] <= threshold) {
          leftIndices.push(j);
        } else {
          rightIndices.push(j);
        }
      }
      
      // Skip if either split is empty
      if (leftIndices.length === 0 || rightIndices.length === 0) continue;
      
      // Calculate information gain
      const leftY = leftIndices.map(idx => y[idx]);
      const rightY = rightIndices.map(idx => y[idx]);
      
      const leftClassCounts = Array(numClasses).fill(0);
      for (const cls of leftY) {
        leftClassCounts[cls]++;
      }
      
      const rightClassCounts = Array(numClasses).fill(0);
      for (const cls of rightY) {
        rightClassCounts[cls]++;
      }
      
      const leftGini = 1 - leftClassCounts.reduce((sum, count) => 
        sum + Math.pow(count / leftY.length, 2), 0);
      
      const rightGini = 1 - rightClassCounts.reduce((sum, count) => 
        sum + Math.pow(count / rightY.length, 2), 0);
      
      // Weighted gini impurity
      const leftWeight = leftY.length / m;
      const rightWeight = rightY.length / m;
      const weightedGini = leftWeight * leftGini + rightWeight * rightGini;
      
      const gain = gini - weightedGini;
      
      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = feature;
        bestThreshold = threshold;
        bestLeftX = leftIndices.map(idx => X[idx]);
        bestLeftY = leftY;
        bestRightX = rightIndices.map(idx => X[idx]);
        bestRightY = rightY;
      }
    }
  }
  
  // If no good split found, create a leaf node
  if (bestGain <= 0) {
    const predictedClass = classCounts.indexOf(Math.max(...classCounts));
    
    return {
      isLeaf: true,
      value: predictedClass,
      classCounts,
      samples: m
    };
  }
  
  // Create a decision node
  const leftChild = buildClassificationTree(
    bestLeftX, bestLeftY, depth + 1, maxDepth, minSamplesSplit, featureNames, numClasses
  );
  
  const rightChild = buildClassificationTree(
    bestRightX, bestRightY, depth + 1, maxDepth, minSamplesSplit, featureNames, numClasses
  );
  
  return {
    isLeaf: false,
    feature: bestFeature,
    featureName: featureNames[bestFeature],
    threshold: bestThreshold,
    gain: bestGain,
    left: leftChild,
    right: rightChild,
    samples: m,
    depth: depth
  };
}

/**
 * Make a prediction using a decision tree
 */
function predictFromTree(sample: number[], tree: any): number {
  if (tree.isLeaf) {
    return tree.value;
  }
  
  if (sample[tree.feature] <= tree.threshold) {
    return predictFromTree(sample, tree.left);
  } else {
    return predictFromTree(sample, tree.right);
  }
}

/**
 * Train a random forest model
 */
export function trainRandomForestModel(trainingData: TrainingData, hyperparameters?: any): ModelResults {
  const { X_train, y_train, X_test, y_test, featureNames, isClassification, classes } = trainingData;
  
  // Get hyperparameters
  const numTrees = hyperparameters?.numTrees || 10;
  const maxDepth = hyperparameters?.maxDepth || 5;
  const bootstrapRatio = hyperparameters?.bootstrapRatio || 0.7;
  
  if (isClassification) {
    return trainRandomForestClassifier(
      X_train, y_train, X_test, y_test, featureNames,
      classes || [], numTrees, maxDepth, bootstrapRatio
    );
  } else {
    return trainRandomForestRegressor(
      X_train, y_train, X_test, y_test, featureNames,
      numTrees, maxDepth, bootstrapRatio
    );
  }
}

/**
 * Train a random forest for regression
 */
function trainRandomForestRegressor(
  X_train: number[][], 
  y_train: number[][], 
  X_test: number[][], 
  y_test: number[][],
  featureNames: string[],
  numTrees: number = 10,
  maxDepth: number = 5,
  bootstrapRatio: number = 0.7
): ModelResults {
  try {
    // For simplicity, we'll use the first target variable if there are multiple
    const y_train_first = y_train.map(row => row[0]);
    
    // Train multiple trees
    const trees = [];
    const featureImportances = Array(featureNames.length).fill(0);
    
    for (let i = 0; i < numTrees; i++) {
      // Bootstrap sampling
      const bootstrapIndices = bootstrapSample(X_train.length, bootstrapRatio);
      const bootstrapX = bootstrapIndices.map(idx => X_train[idx]);
      const bootstrapY = bootstrapIndices.map(idx => y_train_first[idx]);
      
      // Feature subsampling (sqrt of features for each tree)
      const numFeaturesToConsider = Math.max(1, Math.floor(Math.sqrt(featureNames.length)));
      const featureIndices = sampleWithoutReplacement(featureNames.length, numFeaturesToConsider);
      
      // Create subsampled data
      const subsampledX = bootstrapX.map(row => featureIndices.map(idx => row[idx]));
      const subsampledFeatureNames = featureIndices.map(idx => featureNames[idx]);
      
      // Build tree
      const tree = buildRegressionTree(
        subsampledX, 
        bootstrapY, 
        0, 
        maxDepth, 
        2, // minSamplesSplit
        subsampledFeatureNames
      );
      
      // Store tree with feature mapping
      trees.push({
        tree,
        featureMap: featureIndices
      });
      
      // Accumulate feature importance
      accumulateFeatureImportance(tree, featureImportances, featureIndices);
    }
    
    // Make predictions
    const predictions = X_test.map(sample => {
      // Get predictions from all trees
      const treePredictions = trees.map(({ tree, featureMap }) => {
        // Map sample to tree's feature space
        const mappedSample = featureMap.map(idx => sample[idx]);
        return predictFromTree(mappedSample, tree);
      });
      
      // Average predictions
      const avgPrediction = treePredictions.reduce((sum, pred) => sum + pred, 0) / numTrees;
      return [avgPrediction];
    });
    
    // Calculate metrics
    const mse = calculateMSE(predictions, y_test);
    const r2 = calculateR2(predictions, y_test);
    
    // Normalize feature importance
    const totalImportance = featureImportances.reduce((sum, val) => sum + val, 0);
    const normalizedImportance = featureImportances.map(val => 
      totalImportance > 0 ? val / totalImportance : 1 / featureNames.length
    );
    
    // Create feature importance objects
    const featureImportance = featureNames.map((feature, index) => ({
      feature,
      importance: normalizedImportance[index]
    })).sort((a, b) => b.importance - a.importance);
    
    return {
      modelType: 'randomForest',
      modelName: `隨機森林回歸 (樹數=${numTrees}, 深度=${maxDepth})`,
      mse,
      r2,
      featureImportance,
      predictions,
      actual: y_test
    };
  } catch (error) {
    console.error('Error in random forest regression:', error);
    throw error;
  }
}

/**
 * Train a random forest for classification
 */
function trainRandomForestClassifier(
  X_train: number[][], 
  y_train: number[][], 
  X_test: number[][], 
  y_test: number[][],
  featureNames: string[],
  classes: (string | number)[],
  numTrees: number = 15,
  maxDepth: number = 15,
  bootstrapRatio: number = 0.7
): ModelResults {
  try {
    // For classification, we use the first column of y
    const y_train_first = y_train.map(row => row[0]);
    const numClasses = classes.length;
    
    // Train multiple trees
    const trees = [];
    const featureImportances = Array(featureNames.length).fill(0);
    
    for (let i = 0; i < numTrees; i++) {
      // Bootstrap sampling
      const bootstrapIndices = bootstrapSample(X_train.length, bootstrapRatio);
      const bootstrapX = bootstrapIndices.map(idx => X_train[idx]);
      const bootstrapY = bootstrapIndices.map(idx => y_train_first[idx]);
      
      // Feature subsampling (sqrt of features for each tree)
      const numFeaturesToConsider = Math.max(1, Math.floor(Math.sqrt(featureNames.length)));
      const featureIndices = sampleWithoutReplacement(featureNames.length, numFeaturesToConsider);
      
      // Create subsampled data
      const subsampledX = bootstrapX.map(row => featureIndices.map(idx => row[idx]));
      const subsampledFeatureNames = featureIndices.map(idx => featureNames[idx]);
      
      // Build tree
      const tree = buildClassificationTree(
        subsampledX, 
        bootstrapY, 
        0, 
        maxDepth, 
        2, // minSamplesSplit
        subsampledFeatureNames,
        numClasses
      );
      
      // Store tree with feature mapping
      trees.push({
        tree,
        featureMap: featureIndices
      });
      
      // Accumulate feature importance
      accumulateFeatureImportance(tree, featureImportances, featureIndices);
    }
    
    // Make predictions
    const predictions = X_test.map(sample => {
      // Get predictions from all trees
      const treePredictions = trees.map(({ tree, featureMap }) => {
        // Map sample to tree's feature space
        const mappedSample = featureMap.map(idx => sample[idx]);
        return predictFromTree(mappedSample, tree);
      });
      
      // Majority vote
      const votes = Array(numClasses).fill(0);
      treePredictions.forEach(pred => votes[pred]++);
      const predictedClass = votes.indexOf(Math.max(...votes));
      
      return [predictedClass];
    });
    
    // Calculate metrics
    const accuracy = calculateAccuracy(predictions, y_test);
    const { precision, recall, f1Score } = calculateClassificationMetrics(
      predictions, y_test, numClasses
    );
    const confusionMatrix = calculateConfusionMatrix(predictions, y_test, numClasses);
    
    // Normalize feature importance
    const totalImportance = featureImportances.reduce((sum, val) => sum + val, 0);
    const normalizedImportance = featureImportances.map(val => 
      totalImportance > 0 ? val / totalImportance : 1 / featureNames.length
    );
    
    // Create feature importance objects
    const featureImportance = featureNames.map((feature, index) => ({
      feature,
      importance: normalizedImportance[index]
    })).sort((a, b) => b.importance - a.importance);
    
    return {
      modelType: 'randomForest',
      modelName: `隨機森林分類 (樹數=${numTrees}, 深度=${maxDepth})`,
      mse: 0, // Not relevant for classification
      r2: 0,  // Not relevant for classification
      accuracy,
      precision,
      recall,
      f1Score,
      confusionMatrix,
      featureImportance,
      predictions,
      actual: y_test,
      isClassification: true,
      classes
    };
  } catch (error) {
    console.error('Error in random forest classification:', error);
    throw error;
  }
}

/**
 * Bootstrap sampling with replacement
 */
function bootstrapSample(size: number, ratio: number): number[] {
  const sampleSize = Math.floor(size * ratio);
  const indices = [];
  
  for (let i = 0; i < sampleSize; i++) {
    indices.push(Math.floor(Math.random() * size));
  }
  
  return indices;
}

/**
 * Sample without replacement
 */
function sampleWithoutReplacement(size: number, sampleSize: number): number[] {
  const indices = Array.from({ length: size }, (_, i) => i);
  shuffleArray(indices);
  return indices.slice(0, sampleSize);
}

/**
 * Accumulate feature importance from a tree
 */
function accumulateFeatureImportance(
  tree: any, 
  importances: number[], 
  featureMap: number[]
): void {
  if (!tree.isLeaf) {
    // Map local feature index to global feature index
    const globalFeatureIndex = featureMap[tree.feature];
    
    // Add importance (weighted by samples and gain)
    importances[globalFeatureIndex] += tree.gain * tree.samples;
    
    // Recurse
    accumulateFeatureImportance(tree.left, importances, featureMap);
    accumulateFeatureImportance(tree.right, importances, featureMap);
  }
}

/**
 * Normalize features
 */
function normalizeFeatures(X_train: number[][], X_test: number[][]) {
  const n = X_train[0].length; // Number of features
  const m = X_train.length;    // Number of training examples
  
  // Calculate mean and standard deviation for each feature
  const means = Array(n).fill(0);
  const stds = Array(n).fill(0);
  
  // Calculate means
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < m; i++) {
      means[j] += X_train[i][j];
    }
    means[j] /= m;
  }
  
  // Calculate standard deviations
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < m; i++) {
      stds[j] += Math.pow(X_train[i][j] - means[j], 2);
    }
    stds[j] = Math.sqrt(stds[j] / m);
    // Avoid division by zero
    if (stds[j] === 0) stds[j] = 1;
  }
  
  // Normalize training data
  const X_train_norm = X_train.map(row => 
    row.map((val, j) => (val - means[j]) / stds[j])
  );
  
  // Normalize test data using training means and stds
  const X_test_norm = X_test.map(row => 
    row.map((val, j) => (val - means[j]) / stds[j])
  );
  
  return { X_train_norm, X_test_norm, means, stds };
}

/**
 * Calculate mean squared error
 */
export function calculateMSE(predictions: number[][], actual: number[][]): number {
  let sumSquaredError = 0;
  let count = 0;
  
  for (let i = 0; i < predictions.length; i++) {
    for (let j = 0; j < predictions[i].length; j++) {
      if (i < actual.length && j < actual[i].length) {
        sumSquaredError += Math.pow(predictions[i][j] - actual[i][j], 2);
        count++;
      }
    }
  }
  
  return count > 0 ? sumSquaredError / count : 0;
}

/**
 * Calculate R-squared (coefficient of determination)
 */
export function calculateR2(predictions: number[][], actual: number[][]): number {
  // Calculate total sum of squares
  let totalSumSquares = 0;
  let residualSumSquares = 0;
  let count = 0;
  
  // Calculate mean of actual values
  const means = [];
  for (let j = 0; j < actual[0].length; j++) {
    let sum = 0;
    let validCount = 0;
    
    for (let i = 0; i < actual.length; i++) {
      if (j < actual[i].length) {
        sum += actual[i][j];
        validCount++;
      }
    }
    
    means.push(validCount > 0 ? sum / validCount : 0);
  }
  
  // Calculate sums of squares
  for (let i = 0; i < predictions.length; i++) {
    for (let j = 0; j < predictions[i].length; j++) {
      if (i < actual.length && j < actual[i].length) {
        totalSumSquares += Math.pow(actual[i][j] - means[j], 2);
        residualSumSquares += Math.pow(predictions[i][j] - actual[i][j], 2);
        count++;
      }
    }
  }
  
  // Avoid division by zero
  if (totalSumSquares === 0 || count === 0) return 0;
  
  return 1 - (residualSumSquares / totalSumSquares);
}

/**
 * Calculate accuracy for classification
 */
function calculateAccuracy(predictions: number[][], actual: number[][]): number {
  let correct = 0;
  let total = 0;
  
  for (let i = 0; i < predictions.length; i++) {
    if (i < actual.length) {
      // For simplicity, we'll use the first column
      if (predictions[i][0] === actual[i][0]) {
        correct++;
      }
      total++;
    }
  }
  
  return total > 0 ? correct / total : 0;
}

/**
 * Calculate precision, recall, and F1 score for classification
 */
function calculateClassificationMetrics(
  predictions: number[][], 
  actual: number[][],
  numClasses: number
): { precision: number; recall: number; f1Score: number } {
  // For multi-class, we'll calculate macro-averaged metrics
  const truePositives = Array(numClasses).fill(0);
  const falsePositives = Array(numClasses).fill(0);
  const falseNegatives = Array(numClasses).fill(0);
  
  for (let i = 0; i < predictions.length; i++) {
    if (i < actual.length) {
      const predicted = predictions[i][0];
      const truth = actual[i][0];
      
      if (predicted === truth) {
        truePositives[predicted]++;
      } else {
        falsePositives[predicted]++;
        falseNegatives[truth]++;
      }
    }
  }
  
  // Calculate per-class metrics
  const precisions = truePositives.map((tp, i) => 
    tp + falsePositives[i] > 0 ? tp / (tp + falsePositives[i]) : 0
  );
  
  const recalls = truePositives.map((tp, i) => 
    tp + falseNegatives[i] > 0 ? tp / (tp + falseNegatives[i]) : 0
  );
  
  const f1Scores = precisions.map((precision, i) => 
    precision + recalls[i] > 0 ? 2 * precision * recalls[i] / (precision + recalls[i]) : 0
  );
  
  // Calculate macro-averaged metrics
  const macroAvgPrecision = precisions.reduce((sum, val) => sum + val, 0) / numClasses;
  const macroAvgRecall = recalls.reduce((sum, val) => sum + val, 0) / numClasses;
  const macroAvgF1 = f1Scores.reduce((sum, val) => sum + val, 0) / numClasses;
  
  return {
    precision: macroAvgPrecision,
    recall: macroAvgRecall,
    f1Score: macroAvgF1
  };
}

/**
 * Calculate confusion matrix for classification
 */
function calculateConfusionMatrix(
  predictions: number[][], 
  actual: number[][],
  numClasses: number
): number[][] {
  // Initialize confusion matrix
  const confusionMatrix = Array(numClasses).fill(0).map(() => Array(numClasses).fill(0));
  
  for (let i = 0; i < predictions.length; i++) {
    if (i < actual.length) {
      const predicted = predictions[i][0];
      const truth = actual[i][0];
      
      if (predicted < numClasses && truth < numClasses) {
        confusionMatrix[truth][predicted]++;
      }
    }
  }
  
  return confusionMatrix;
}

/**
 * Calculate feature importance for linear models
 */
function calculateLinearFeatureImportance(
  weights: number[][], 
  featureNames: string[]
): { feature: string; importance: number }[] {
  const n = featureNames.length;
  const numOutputs = weights.length;
  
  // Calculate absolute weights for each feature
  const importanceValues = Array(n).fill(0);
  
  for (let j = 0; j < n; j++) {
    for (let k = 0; k < numOutputs; k++) {
      importanceValues[j] += Math.abs(weights[k][j]);
    }
    // Average across outputs
    importanceValues[j] /= numOutputs;
  }
  
  // Normalize
  const totalImportance = importanceValues.reduce((sum, val) => sum + val, 0);
  const normalizedImportance = importanceValues.map(val => 
    totalImportance > 0 ? val / totalImportance : 1 / n
  );
  
  // Create feature importance objects
  return featureNames.map((feature, index) => ({
    feature,
    importance: normalizedImportance[index]
  })).sort((a, b) => b.importance - a.importance);
}

/**
 * Calculate feature importance for tree-based models
 */
function calculateTreeFeatureImportance(
  tree: any, 
  featureNames: string[]
): { feature: string; importance: number }[] {
  // Initialize importance values
  const importanceValues = Array(featureNames.length).fill(0);
  
  // Calculate total samples in the tree
  const totalSamples = tree.samples;
  
  // Recursively calculate importance
  calculateNodeImportance(tree, importanceValues, totalSamples);
  
  // Normalize
  const totalImportance = importanceValues.reduce((sum, val) => sum + val, 0);
  const normalizedImportance = importanceValues.map(val => 
    totalImportance > 0 ? val / totalImportance : 1 / featureNames.length
  );
  
  // Create feature importance objects
  return featureNames.map((feature, index) => ({
    feature,
    importance: normalizedImportance[index]
  })).sort((a, b) => b.importance - a.importance);
}

/**
 * Recursively calculate node importance
 */
function calculateNodeImportance(
  node: any , 
  importances: number[], 
  totalSamples: number
): void {
  if (!node.isLeaf) {
    // Add importance (weighted by samples and gain)
    importances[node.feature] += node.gain * (node.samples / totalSamples);
    
    // Recurse
    calculateNodeImportance(node.left, importances, totalSamples);
    calculateNodeImportance(node.right, importances, totalSamples);
  }
}

/**
 * Shuffle array in-place
 */
export function shuffleArray(array: any[]): void {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

/**
 * Get model trainer function by type
 */
export function getModelTrainer(modelType: ModelType): (trainingData: TrainingData, hyperparameters?: any) => ModelResults {
  switch (modelType) {
    case 'linear':
      return trainLinearModel;
    case 'decisionTree':
      return trainDecisionTreeModel;
    case 'randomForest':
      return trainRandomForestModel;
    default:
      throw new Error(`Unknown model type: ${modelType}`);
  }
}

/**
 * Get available models
 */
export function getAvailableModels(): { type: ModelType; name: string; description: string }[] {
  return [
    {
      type: 'linear',
      name: '線性模型',
      description: '適用於線性關係，可用於回歸或分類 (邏輯迴歸)'
    },
    {
      type: 'decisionTree',
      name: '決策樹',
      description: '樹狀結構模型，可捕捉非線性關係，易於解釋'
    },
    {
      type: 'randomForest',
      name: '隨機森林',
      description: '多棵決策樹的集成模型，通常比單一決策樹效能更好'
    }
  ];
}