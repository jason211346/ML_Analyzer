import React, { useState } from 'react';
import { ModelResults as ModelResultsType } from '../types';
import { Bar, Line, Pie } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

interface ModelResultsProps {
  results: ModelResultsType | null;
}

const ModelResults: React.FC<ModelResultsProps> = ({ results }) => {
  const [showAllPredictions, setShowAllPredictions] = useState(false);

  if (!results) {
    return null;
  }

  const isClassification = results.isClassification || false;

  // Feature importance chart data
  const featureImportanceData = {
    labels: results.featureImportance.map(item => item.feature),
    datasets: [
      {
        label: '特徵重要性',
        data: results.featureImportance.map(item => item.importance * 100), // Convert to percentage
        backgroundColor: 'rgba(54, 162, 235, 0.6)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1
      }
    ]
  };

  // Prepare prediction vs actual data
  // For simplicity, we'll just use the first target variable if there are multiple
  const actualValues = results.actual.map(row => row[0]);
  const predictedValues = results.predictions.map(row => row[0]);
  
  // For classification, prepare confusion matrix visualization
  let confusionMatrixData = null;
  if (isClassification && results.confusionMatrix) {
    const classLabels = results.classes?.map((c, i) => `類別 ${i}: ${c}`) || 
                        Array.from({ length: results.confusionMatrix.length }, (_, i) => `類別 ${i}`);
    
    confusionMatrixData = {
      labels: classLabels,
      datasets: classLabels.map((label, i) => ({
        label,
        data: results.confusionMatrix![i],
        backgroundColor: `rgba(${54 + i * 50}, ${162 - i * 20}, ${235 - i * 30}, 0.6)`,
        borderColor: `rgba(${54 + i * 50}, ${162 - i * 20}, ${235 - i * 30}, 1)`,
        borderWidth: 1
      }))
    };
  }
  
  // For regression, prepare prediction vs actual chart
  let predictionData = null;
  if (!isClassification) {
    predictionData = {
      labels: Array.from({ length: actualValues.length }, (_, i) => `樣本 ${i + 1}`),
      datasets: [
        {
          label: '實際值',
          data: actualValues,
          borderColor: 'rgba(75, 192, 192, 1)',
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          pointRadius: 4,
          tension: 0.1
        },
        {
          label: '預測值',
          data: predictedValues,
          borderColor: 'rgba(255, 99, 132, 1)',
          backgroundColor: 'rgba(255, 99, 132, 0.2)',
          pointRadius: 4,
          tension: 0.1
        }
      ]
    };
  } else {
    // For classification, create a different visualization
    const uniqueClasses = [...new Set(actualValues)].sort((a, b) => a - b);
    const classLabels = results.classes || uniqueClasses.map(c => `類別 ${c}`);
    
    // Count predictions per class
    const actualCounts = uniqueClasses.map(classVal => 
      actualValues.filter(val => val === classVal).length
    );
    
    const predictedCounts = uniqueClasses.map(classVal => 
      predictedValues.filter(val => val === classVal).length
    );
    
    predictionData = {
      labels: classLabels,
      datasets: [
        {
          label: '實際分佈',
          data: actualCounts,
          backgroundColor: 'rgba(75, 192, 192, 0.6)',
          borderColor: 'rgba(75, 192, 192, 1)',
          borderWidth: 1
        },
        {
          label: '預測分佈',
          data: predictedCounts,
          backgroundColor: 'rgba(255, 99, 132, 0.6)',
          borderColor: 'rgba(255, 99, 132, 1)',
          borderWidth: 1
        }
      ]
    };
  }

  // Calculate additional metrics
  const meanActual = actualValues.reduce((sum, val) => sum + val, 0) / actualValues.length;
  const rmse = Math.sqrt(results.mse);
  const mae = actualValues.reduce((sum, val, i) => sum + Math.abs(val - predictedValues[i]), 0) / actualValues.length;
  
  // Determine how many predictions to show in the table
  const displayPredictions = showAllPredictions 
    ? results.predictions 
    : results.predictions.slice(0, 10);
  const displayActual = showAllPredictions 
    ? results.actual 
    : results.actual.slice(0, 10);

  return (
    <div className="bg-white rounded-lg shadow p-6 mb-6">
      <h2 className="text-xl font-semibold mb-2">模型分析結果</h2>
      <div className="text-sm text-gray-600 mb-6">
        使用模型: <span className="font-medium">{results.modelName}</span>
        {isClassification && <span className="ml-2">(分類任務)</span>}
        {!isClassification && <span className="ml-2">(回歸任務)</span>}
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        {!isClassification ? (
          // Regression metrics
          <>
            <div className="bg-blue-50 p-4 rounded-lg">
              <h3 className="text-sm font-medium mb-1">均方誤差 (MSE)</h3>
              <p className="text-2xl font-bold">{results.mse.toFixed(4)}</p>
              <p className="text-xs text-gray-600 mt-1">越低表示預測誤差越小</p>
            </div>
            
            <div className="bg-purple-50 p-4 rounded-lg">
              <h3 className="text-sm font-medium mb-1">均方根誤差 (RMSE)</h3>
              <p className="text-2xl font-bold">{rmse.toFixed(4)}</p>
              <p className="text-xs text-gray-600 mt-1">與原始資料單位相同的誤差</p>
            </div>
            
            <div className="bg-indigo-50 p-4 rounded-lg">
              <h3 className="text-sm font-medium mb-1">平均絕對誤差 (MAE)</h3>
              <p className="text-2xl font-bold">{mae.toFixed(4)}</p>
              <p className="text-xs text-gray-600 mt-1">預測值與實際值的平均差距</p>
            </div>
            
            <div className="bg-green-50 p-4 rounded-lg">
              <h3 className="text-sm font-medium mb-1">決定係數 (R²)</h3>
              <p className="text-2xl font-bold">{results.r2.toFixed(4)}</p>
              <p className="text-xs text-gray-600 mt-1">越接近1表示模型解釋力越強</p>
            </div>
          </>
        ) : (
          // Classification metrics
          <>
            <div className="bg-blue-50 p-4 rounded-lg">
              <h3 className="text-sm font-medium mb-1">準確率 (Accuracy)</h3>
              <p className="text-2xl font-bold">{results.accuracy !== undefined ? (results.accuracy * 100).toFixed(2) + '%' : 'N/A'}</p>
              <p className="text-xs text-gray-600 mt-1">正確預測的比例</p>
            </div>
            
            <div className="bg-purple-50 p-4 rounded-lg">
              <h3 className="text-sm font-medium mb-1">精確率 (Precision)</h3>
              <p className="text-2xl font-bold">{results.precision !== undefined ? (results.precision * 100).toFixed(2) + '%' : 'N/A'}</p>
              <p className="text-xs text-gray-600 mt-1">預測為正例中實際為正例的比例</p>
            </div>
            
            <div className="bg-indigo-50 p-4 rounded-lg">
              <h3 className="text-sm font-medium mb-1">召回率 (Recall)</h3>
              <p className="text-2xl font-bold">{results.recall !== undefined ? (results.recall * 100).toFixed(2) + '%' : 'N/A'}</p>
              <p className="text-xs text-gray-600 mt-1">實際為正例中被正確預測的比例</p>
            </div>
            
            <div className="bg-green-50 p-4 rounded-lg">
              <h3 className="text-sm font-medium mb-1">F1 分數</h3>
              <p className="text-2xl font-bold">{results.f1Score !== undefined ? (results.f1Score * 100).toFixed(2) + '%' : 'N/A'}</p>
              <p className="text-xs text-gray-600 mt-1">精確率和召回率的調和平均</p>
            </div>
          </>
        )}
      </div>
      
      <div className="mb-8">
        <h3 className="text-lg font-medium mb-4">特徵重要性分析</h3>
        <div className="h-80">
          <Bar
            data={featureImportanceData}
            options={{
              indexAxis: 'y',
              plugins: {
                title: {
                  display: true,
                  text: '特徵重要性 (%)'
                },
                legend: {
                  display: false
                },
                tooltip: {
                  callbacks: {
                    label: function(context) {
                      return `${context.parsed.x.toFixed(2)}%`;
                    }
                  }
                }
              },
              scales: {
                x: {
                  beginAtZero: true,
                  title: {
                    display: true,
                    text: '重要性 (%)'
                  }
                }
              },
              maintainAspectRatio: false
            }}
          />
        </div>
      </div>
      
      <div className="mb-8">
        <h3 className="text-lg font-medium mb-4">預測結果分析</h3>
        <div className="h-80">
          {!isClassification ? (
            // Regression: Line chart for actual vs predicted
            <Line
              data={predictionData}
              options={{
                plugins: {
                  title: {
                    display: true,
                    text: '測試集預測結果'
                  }
                },
                scales: {
                  y: {
                    title: {
                      display: true,
                      text: '值'
                    }
                  },
                  x: {
                    title: {
                      display: true,
                      text: '測試樣本'
                    },
                    ticks: {
                      maxTicksLimit: 10
                    }
                  }
                },
                maintainAspectRatio: false
              }}
            />
          ) : (
            // Classification: Bar chart for class distribution
            <Bar
              data={predictionData}
              options={{
                plugins: {
                  title: {
                    display: true,
                    text: '類別分佈比較'
                  }
                },
                scales: {
                  y: {
                    beginAtZero: true,
                    title: {
                      display: true,
                      text: '樣本數'
                    }
                  },
                  x: {
                    title: {
                      display: true,
                      text: '類別'
                    }
                  }
                },
                maintainAspectRatio: false
              }}
            />
          )}
        </div>
      </div>
      
      {isClassification && results.confusionMatrix && (
        <div className="mb-8">
          <h3 className="text-lg font-medium mb-4">混淆矩陣</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full border border-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">實際\預測</th>
                  {(results.classes || Array.from({ length: results.confusionMatrix.length }, (_, i) => i)).map((cls, i) => (
                    <th key={i} className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      類別 {cls}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {results.confusionMatrix.map((row, i) => (
                  <tr key={i}>
                    <td className="px-4 py-2 whitespace-nowrap font-medium">類別 {results.classes ? results.classes[i] : i}</td>
                    {row.map((cell, j) => (
                      <td key={j} className={`px-4 py-2 whitespace-nowrap ${i === j ? 'bg-green-50 font-medium' : ''}`}>
                        {cell}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            * 對角線上的數值表示正確分類的樣本數
          </p>
        </div>
      )}
      
      <div>
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-medium">預測詳細資料</h3>
          <button
            onClick={() => setShowAllPredictions(!showAllPredictions)}
            className="text-sm text-blue-600 hover:text-blue-800"
          >
            {showAllPredictions ? '顯示前10筆' : '顯示全部'}
          </button>
        </div>
        
        <div className="overflow-x-auto">
          <table className="min-w-full border border-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">樣本</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">實際值</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">預測值</th>
                {!isClassification && (
                  <>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">誤差</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">誤差百分比</th>
                  </>
                )}
                {isClassification && (
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">預測結果</th>
                )}
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {displayPredictions.map((prediction, index) => {
                const actual = displayActual[index][0];
                const predicted = prediction[0];
                
                if (isClassification) {
                  // For classification
                  const isCorrect = actual === predicted;
                  const actualClass = results.classes ? results.classes[actual] : actual;
                  const predictedClass = results.classes ? results.classes[predicted] : predicted;
                  
                  return (
                    <tr key={index}>
                      <td className="px-4 py-2 whitespace-nowrap">樣本 {index + 1}</td>
                      <td className="px-4 py-2 whitespace-nowrap">{actualClass}</td>
                      <td className="px-4 py-2 whitespace-nowrap">{predictedClass}</td>
                      <td className="px-4 py-2 whitespace-nowrap">
                        <span className={isCorrect ? 'text-green-600' : 'text-red-600'}>
                          {isCorrect ? '正確' : '錯誤'}
                        </span>
                      </td>
                    </tr>
                  );
                } else {
                  // For regression
                  const error = predicted - actual;
                  const percentError = actual !== 0 ? (error / Math.abs(actual)) * 100 : 0;
                  
                  return (
                    <tr key={index}>
                      <td className="px-4 py-2 whitespace-nowrap">樣本 {index + 1}</td>
                      <td className="px-4 py-2 whitespace-nowrap">{actual.toFixed(4)}</td>
                      <td className="px-4 py-2 whitespace-nowrap">{predicted.toFixed(4)}</td>
                      <td className="px-4 py-2 whitespace-nowrap">
                        <span className={error < 0 ? 'text-green-600' : 'text-red-600'}>
                          {error.toFixed(4)}
                        </span>
                      </td>
                      <td className="px-4 py-2 whitespace-nowrap">
                        <span className={Math.abs(percentError) < 10 ? 'text-green-600' : 'text-red-600'}>
                          {percentError.toFixed(2)}%
                        </span>
                      </td>
                    </tr>
                  );
                }
              })}
            </tbody>
          </table>
        </div>
        
        {!showAllPredictions && results.predictions.length > 10 && (
          <div className="mt-2 text-sm text-gray-500 text-right">
            顯示 10 / {results.predictions.length} 筆資料
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelResults;