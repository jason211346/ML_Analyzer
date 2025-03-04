import React, { useState } from 'react';
import { ExcelData, FeatureSelection, ModelResults, ModelType, ModelConfig, ModelComparisonResults, TaskType } from '../types';
import { splitData, trainModel, crossValidate, compareModels } from '../utils/mlUtils';
import { getAvailableModels } from '../utils/mlModels';
import { BarChart, LineChart, ArrowRight, AlertTriangle, Info, Check, RefreshCw, BarChart2, PieChart } from 'lucide-react';
import toast from 'react-hot-toast';

interface ModelTrainerProps {
  data: ExcelData | null;
  featureSelection: FeatureSelection;
  onModelTrained: (results: ModelResults) => void;
}

const ModelTrainer: React.FC<ModelTrainerProps> = ({
  data,
  featureSelection,
  onModelTrained
}) => {
  const [isTraining, setIsTraining] = useState(false);
  const [validationResults, setValidationResults] = useState<{ avgMSE: number; avgR2: number; avgAccuracy?: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dataStats, setDataStats] = useState<{
    totalRows: number;
    validRows: number;
    numericFeatures: string[];
    uniqueValues: Map<string, Set<any>>;
  } | null>(null);
  const [selectedModelType, setSelectedModelType] = useState<ModelType>('linear');
  const [compareAllModels, setCompareAllModels] = useState(false);
  const [comparisonResults, setComparisonResults] = useState<ModelComparisonResults | null>(null);
  const [taskType, setTaskType] = useState<TaskType>('regression');
  const [advancedOptions, setAdvancedOptions] = useState(false);
  const [hyperparameters, setHyperparameters] = useState({
    linear: {
      regularization: 0.01,
      learningRate: 0.01,
      iterations: 100
    },
    decisionTree: {
      maxDepth: 5,
      minSamplesSplit: 2
    },
    randomForest: {
      numTrees: 15,
      maxDepth: 8,
      bootstrapRatio: 0.8
    }
  });

  // Get available models
  const availableModels = getAvailableModels();

  // Analyze data when features are selected
  React.useEffect(() => {
    if (!data || featureSelection.inputFeatures.length === 0) {
      setDataStats(null);
      return;
    }

    try {
      // Get indices of selected features
      const allFeatures = [...featureSelection.inputFeatures, ...featureSelection.targetFeatures];
      const featureIndices = allFeatures.map(feature => data.headers.indexOf(feature));
      
      // Count valid rows
      let validRowCount = 0;
      const numericValues = new Set<string>();
      const uniqueValues = new Map<string, Set<any>>();
      
      // Initialize unique values map
      allFeatures.forEach(feature => {
        uniqueValues.set(feature, new Set());
      });
      
      data.data.forEach(row => {
        // Skip rows that are too short
        if (row.length < Math.max(...featureIndices.filter(idx => idx !== -1)) + 1) {
          return;
        }
        
        const isValid = featureIndices.every(index => {
          if (index === -1) return false;
          if (index >= row.length) return false;
          
          const value = row[index];
          return value !== undefined && value !== null;
        });
        
        if (isValid) {
          validRowCount++;
          
          // Track which features have numeric values and collect unique values
          featureIndices.forEach((index, i) => {
            if (index !== -1 && index < row.length) {
              const feature = allFeatures[i];
              const value = row[index];
              
              // Add to unique values
              uniqueValues.get(feature)?.add(value);
              
              // Check if numeric
              const numValue = typeof value === 'string' ? Number(value) : value;
              if (!isNaN(numValue)) {
                numericValues.add(allFeatures[i]);
              }
            }
          });
        }
      });
      
      setDataStats({
        totalRows: data.data.length,
        validRows: validRowCount,
        numericFeatures: Array.from(numericValues),
        uniqueValues
      });
    } catch (err) {
      console.error("Error analyzing data:", err);
      setDataStats(null);
    }
  }, [data, featureSelection]);

  // Detect if target is likely classification
  React.useEffect(() => {
    if (dataStats && featureSelection.targetFeatures.length > 0) {
      const targetFeature = featureSelection.targetFeatures[0];
      const uniqueCount = dataStats.uniqueValues.get(targetFeature)?.size || 0;
      
      // If target has few unique values (<=10), suggest classification
      if (uniqueCount > 0 && uniqueCount <= 10) {
        setTaskType('classification');
      } else {
        setTaskType('regression');
      }
    }
  }, [dataStats, featureSelection]);

  const handleTaskTypeChange = (type: TaskType) => {
    setTaskType(type);
  };

  const handleHyperparameterChange = (model: ModelType, param: string, value: any) => {
    setHyperparameters(prev => ({
      ...prev,
      [model]: {
        ...prev[model],
        [param]: value
      }
    }));
  };

  const handleTrainModel = async () => {
    if (!data) {
      toast.error('請先上傳Excel檔案');
      return;
    }

    if (featureSelection.inputFeatures.length === 0) {
      toast.error('請至少選擇一個輸入特徵 (X)');
      return;
    }

    if (featureSelection.targetFeatures.length === 0) {
      toast.error('請至少選擇一個預測目標 (Y)');
      return;
    }

    setIsTraining(true);
    setError(null);
    setComparisonResults(null);
    
    try {
      // Split data
      const trainingData = splitData(data, featureSelection, 0.2, taskType);
      
      if (compareAllModels) {
        // Train and compare all models
        const results = compareModels(trainingData, ['linear', 'decisionTree', 'randomForest'], hyperparameters);
        setComparisonResults(results);
        
        // Find the best model and use it
        const bestModel = results.models.find(model => model.modelType === results.bestModel);
        if (bestModel) {
          onModelTrained(bestModel);
          toast.success(`模型比較完成，最佳模型: ${bestModel.modelName}`);
        } else {
          // Use the first model if best model not found
          onModelTrained(results.models[0]);
          toast.success('模型比較完成');
        }
      } else {
        // Perform cross-validation for selected model
        const cvResults = crossValidate(data, featureSelection, selectedModelType, 5, taskType, hyperparameters);
        setValidationResults(cvResults);
        
        // Train selected model
        const results = trainModel(trainingData, selectedModelType, hyperparameters);
        
        onModelTrained(results);
        toast.success('模型訓練完成');
      }
    } catch (error) {
      console.error('Model training error:', error);
      const errorMessage = error instanceof Error ? error.message : '未知錯誤';
      setError(errorMessage);
      toast.error(`模型訓練失敗: ${errorMessage}`);
    } finally {
      setIsTraining(false);
    }
  };

  const isDataValid = data && data.data.length > 0;
  const hasFeatureSelection = featureSelection.inputFeatures.length > 0 && featureSelection.targetFeatures.length > 0;
  const hasEnoughData = dataStats && dataStats.validRows >= 10; // Minimum data requirement
  
  // Check if all selected features are numeric
  const allFeaturesNumeric = dataStats && [...featureSelection.inputFeatures, ...featureSelection.targetFeatures].every(
    feature => dataStats.numericFeatures.includes(feature)
  );

  // Get unique values count for target feature
  const targetUniqueCount = dataStats && featureSelection.targetFeatures.length > 0 
    ? dataStats.uniqueValues.get(featureSelection.targetFeatures[0])?.size || 0
    : 0;

  return (
    <div className="bg-white rounded-lg shadow p-6 mb-6">
      <h2 className="text-xl font-semibold mb-4">模型訓練</h2>
      
      <div className="space-y-4">
        <div className="flex items-center space-x-2">
          <div className={`w-4 h-4 rounded-full ${isDataValid ? 'bg-green-500' : 'bg-red-500'}`}></div>
          <span>資料狀態: {isDataValid ? '有效' : '無效'}</span>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className={`w-4 h-4 rounded-full ${hasFeatureSelection ? 'bg-green-500' : 'bg-yellow-500'}`}></div>
          <span>特徵選擇: {hasFeatureSelection ? '已選擇' : '未完成'}</span>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className={`w-4 h-4 rounded-full ${hasEnoughData ? 'bg-green-500' : 'bg-yellow-500'}`}></div>
          <span>資料量: {dataStats ? `${dataStats.validRows} 筆有效 (共 ${dataStats.totalRows} 筆)` : '無資料'} {!hasEnoughData && '(建議至少10筆)'}</span>
        </div>
        
        {targetUniqueCount > 0 && (
          <div className="flex items-center space-x-2">
            <div className={`w-4 h-4 rounded-full bg-blue-500`}></div>
            <span>目標變數唯一值數量: {targetUniqueCount} {targetUniqueCount <= 10 ? '(適合分類任務)' : '(適合回歸任務)'}</span>
          </div>
        )}
        
        <div className="p-4 bg-gray-50 rounded-lg">
          <h3 className="text-lg font-medium mb-3">任務類型</h3>
          <div className="flex space-x-4 mb-4">
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="radio"
                checked={taskType === 'regression'}
                onChange={() => handleTaskTypeChange('regression')}
                className="h-4 w-4 text-blue-600 rounded focus:ring-blue-500"
              />
              <div className="flex items-center">
                <LineChart size={18} className="mr-1 text-blue-500" />
                <span>回歸 (預測連續值)</span>
              </div>
            </label>
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="radio"
                checked={taskType === 'classification'}
                onChange={() => handleTaskTypeChange('classification')}
                className="h-4 w-4 text-blue-600 rounded focus:ring-blue-500"
              />
              <div className="flex items-center">
                <PieChart size={18} className="mr-1 text-purple-500" />
                <span>分類 (預測類別)</span>
              </div>
            </label>
          </div>
          
          <h3 className="text-lg font-medium mb-3">選擇模型</h3>
          
          <div className="mb-3">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={compareAllModels}
                onChange={(e) => setCompareAllModels(e.target.checked)}
                className="h-4 w-4 text-blue-600 rounded focus:ring-blue-500"
              />
              <span>比較所有模型並選擇最佳模型</span>
            </label>
          </div>
          
          {!compareAllModels && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {availableModels.map((model) => (
                <div 
                  key={model.type}
                  className={`p-3 border rounded-lg cursor-pointer ${
                    selectedModelType === model.type 
                      ? 'border-blue-500 bg-blue-50' 
                      : 'border-gray-200 hover:bg-gray-50'
                  }`}
                  onClick={() => setSelectedModelType(model.type)}
                >
                  <div className="flex items-center justify-between">
                    <h4 className="font-medium">{model.name}</h4>
                    {selectedModelType === model.type && (
                      <Check size={16} className="text-blue-500" />
                    )}
                  </div>
                  <p className="text-sm text-gray-600 mt-1">{model.description}</p>
                </div>
              ))}
            </div>
          )}
          
          <div className="mt-4">
            <button
              onClick={() => setAdvancedOptions(!advancedOptions)}
              className="text-sm text-blue-600 hover:text-blue-800 flex items-center"
            >
              {advancedOptions ? '隱藏進階選項' : '顯示進階選項'}
            </button>
            
            {advancedOptions && (
              <div className="mt-3 border rounded-lg p-4 bg-white">
                <h4 className="font-medium mb-3">模型超參數設定</h4>
                
                {selectedModelType === 'linear' && (
                  <div className="space-y-3">
                    <div>
                      <label className="block text-sm text-gray-700 mb-1">正則化係數 (L2)</label>
                      <input
                        type="number"
                        min="0"
                        step="0.001"
                        value={hyperparameters.linear.regularization}
                        onChange={(e) => handleHyperparameterChange('linear', 'regularization', parseFloat(e.target.value))}
                        className="w-full p-2 border rounded"
                      />
                      <p className="text-xs text-gray-500 mt-1">較大的值可以減少過擬合 (建議: 0.001-0.1)</p>
                    </div>
                    <div>
                      <label className="block text-sm text-gray-700 mb-1">學習率</label>
                      <input
                        type="number"
                        min="0.001"
                        max="1"
                        step="0.001"
                        value={hyperparameters.linear.learningRate}
                        onChange={(e) => handleHyperparameterChange('linear', 'learningRate', parseFloat(e.target.value))}
                        className="w-full p-2 border rounded"
                      />
                      <p className="text-xs text-gray-500 mt-1">控制每次迭代的步長 (建議: 0.001-0.1)</p>
                    </div>
                    <div>
                      <label className="block text-sm text-gray-700 mb-1">迭代次數</label>
                      <input
                        type="number"
                        min="10"
                        max="1000"
                        step="10"
                        value={hyperparameters.linear.iterations}
                        onChange={(e) => handleHyperparameterChange('linear', 'iterations', parseInt(e.target.value))}
                        className="w-full p-2 border rounded"
                      />
                      <p className="text-xs text-gray-500 mt-1">較多的迭代可能提高精度，但訓練時間更長 (建議: 50-500)</p>
                    </div>
                  </div>
                )}
                
                {selectedModelType === 'decisionTree' && (
                  <div className="space-y-3">
                    <div>
                      <label className="block text-sm text-gray-700 mb-1">最大深度</label>
                      <input
                        type="number"
                        min="1"
                        max="20"
                        step="1"
                        value={hyperparameters.decisionTree.maxDepth}
                        onChange={(e) => handleHyperparameterChange('decisionTree', 'maxDepth', parseInt(e.target.value))}
                        className="w-full p-2 border rounded"
                      />
                      <p className="text-xs text-gray-500 mt-1">樹的最大深度，較大的值可能導致過擬合 (建議: 3-10)</p>
                    </div>
                    <div>
                      <label className="block text-sm text-gray-700 mb-1">最小分割樣本數</label>
                      <input
                        type="number"
                        min="2"
                        max="20"
                        step="1"
                        value={hyperparameters.decisionTree.minSamplesSplit}
                        onChange={(e) => handleHyperparameterChange('decisionTree', 'minSamplesSplit', parseInt(e.target.value))}
                        className="w-full p-2 border rounded"
                      />
                      <p className="text-xs text-gray-500 mt-1">分割節點所需的最小樣本數 (建議: 2-10)</p>
                    </div>
                  </div>
                )}
                
                {selectedModelType === 'randomForest' && (
                  <div className="space-y-3">
                    <div>
                      <label className="block text-sm text-gray-700 mb-1">樹的數量</label>
                      <input
                        type="number"
                        min="5"
                        max="100"
                        step="5"
                        value={hyperparameters.randomForest.numTrees}
                        onChange={(e) => handleHyperparameterChange('randomForest', 'numTrees', parseInt(e.target.value))}
                        className="w-full p-2 border rounded"
                      />
                      <p className="text-xs text-gray-500 mt-1">森林中樹的數量，較多的樹通常提高精度但增加計算成本 (建議: 10-50)</p>
                    </div>
                    <div>
                      <label className="block text-sm text-gray-700 mb-1">最大深度</label>
                      <input
                        type="number"
                        min="3"
                        max="20"
                        step="1"
                        value={hyperparameters.randomForest.maxDepth}
                        onChange={(e) => handleHyperparameterChange('randomForest', 'maxDepth', parseInt(e.target.value))}
                        className="w-full p-2 border rounded"
                       />
                      <p className="text-xs text-gray-500 mt-1">每棵樹的最大深度 (建議: 5-15)</p>
                    </div>
                    <div>
                      <label className="block text-sm text-gray-700 mb-1">抽樣比例</label>
                      <input
                        type="number"
                        min="0.1"
                        max="1"
                        step="0.1"
                        value={hyperparameters.randomForest.bootstrapRatio}
                        onChange={(e) => handleHyperparameterChange('randomForest', 'bootstrapRatio', parseFloat(e.target.value))}
                        className="w-full p-2 border rounded"
                      />
                      <p className="text-xs text-gray-500 mt-1">每棵樹使用的訓練資料比例 (建議: 0.6-0.9)</p>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
        
        {dataStats && !allFeaturesNumeric && (
          <div className="flex items-center p-3 bg-yellow-50 text-yellow-800 rounded-lg">
            <AlertTriangle size={18} className="mr-2 flex-shrink-0" />
            <span className="text-sm">
              部分選擇的特徵可能包含非數值資料，這可能導致模型訓練失敗
            </span>
          </div>
        )}
        
        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            <h3 className="font-medium mb-1">錯誤訊息</h3>
            <p className="text-sm">{error}</p>
            <div className="text-sm mt-2">
              請檢查：
              <div className="ml-5 mt-1">
                • 所選特徵是否包含非數值資料
                <br />
                • 資料是否有缺失值
                <br />
                • 特徵名稱是否正確
                <br />
                • 資料格式是否一致
              </div>
            </div>
          </div>
        )}
        
        {validationResults && !compareAllModels && (
          <div className="mt-4 p-4 bg-blue-50 rounded-lg">
            <h3 className="font-medium mb-2">交叉驗證結果 (5-fold)</h3>
            <div className="grid grid-cols-2 gap-4">
              {taskType === 'regression' && (
                <>
                  <div>
                    <p className="text-sm text-gray-600">平均均方誤差 (MSE)</p>
                    <p className="font-semibold">{validationResults.avgMSE.toFixed(4)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">平均決定係數 (R²)</p>
                    <p className="font-semibold">{validationResults.avgR2.toFixed(4)}</p>
                  </div>
                </>
              )}
              {taskType === 'classification' && validationResults.avgAccuracy !== undefined && (
                <div>
                  <p className="text-sm text-gray-600">平均準確率 (Accuracy)</p>
                  <p className="font-semibold">{(validationResults.avgAccuracy * 100).toFixed(2)}%</p>
                </div>
              )}
            </div>
          </div>
        )}
        
        {comparisonResults && (
          <div className="mt-4 p-4 bg-blue-50 rounded-lg">
            <h3 className="font-medium mb-2">模型比較結果</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-2 pr-4">模型</th>
                    {taskType === 'regression' && (
                      <>
                        <th className="text-left py-2 pr-4">MSE</th>
                        <th className="text-left py-2 pr-4">R²</th>
                      </>
                    )}
                    {taskType === 'classification' && (
                      <th className="text-left py-2 pr-4">準確率</th>
                    )}
                    <th className="text-left py-2">最佳模型</th>
                  </tr>
                </thead>
                <tbody>
                  {comparisonResults.models.map((model, index) => (
                    <tr key={index} className={model.modelType === comparisonResults.bestModel ? 'bg-blue-100' : ''}>
                      <td className="py-2 pr-4 font-medium">{model.modelName}</td>
                      {taskType === 'regression' && (
                        <>
                          <td className="py-2 pr-4">{model.mse.toFixed(4)}</td>
                          <td className="py-2 pr-4">{model.r2.toFixed(4)}</td>
                        </>
                      )}
                      {taskType === 'classification' && (
                        <td className="py-2 pr-4">{model.accuracy !== undefined ? (model.accuracy * 100).toFixed(2) + '%' : 'N/A'}</td>
                      )}
                      <td className="py-2">
                        {model.modelType === comparisonResults.bestModel && (
                          <Check size={16} className="text-green-500" />
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
        
        {!hasEnoughData && dataStats && dataStats.validRows > 0 && (
          <div className="flex items-center p-3 bg-yellow-50 text-yellow-800 rounded-lg">
            <AlertTriangle size={18} className="mr-2" />
            <span className="text-sm">資料量較少，模型效能可能不佳</span>
          </div>
        )}
        
        <div className="p-4 bg-blue-50 rounded-lg">
          <div className="flex items-center mb-2">
            <Info size={18} className="text-blue-500 mr-2" />
            <h3 className="font-medium">資料處理說明</h3>
          </div>
          <div className="text-sm space-y-1 text-gray-700">
            • 系統會自動過濾包含非數值或缺失值的資料列
            <br />
            • 將自動分割為訓練集(80%)和測試集(20%)
            <br />
            • 使用交叉驗證評估模型穩定性
            <br />
            • 支援多種機器學習模型，可比較效能選擇最佳模型
            <br />
            • 可以選擇回歸任務(預測連續值)或分類任務(預測類別)
          </div>
        </div>
        
        <div className="flex items-center justify-between">
          <div className="flex items-center text-sm text-gray-500">
            {taskType === 'regression' ? (
              <LineChart size={16} className="mr-1" />
            ) : (
              <BarChart2 size={16} className="mr-1" />
            )}
            <span>點擊開始訓練按鈕執行模型訓練</span>
          </div>
          
          <button
            onClick={handleTrainModel}
            disabled={isTraining || !isDataValid || !hasFeatureSelection}
            className={`px-4 py-2 rounded flex items-center ${
              isTraining || !isDataValid || !hasFeatureSelection
                ? 'bg-gray-300 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-500 text-white'
            }`}
          >
            {isTraining ? (
              <>
                <RefreshCw size={16} className="animate-spin mr-2" />
                訓練中...
              </>
            ) : (
              <>
                開始訓練
                <ArrowRight size={16} className="ml-1" />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ModelTrainer;