import React, { useState, useEffect } from 'react';
import { ArrowUpDown, Search } from 'lucide-react';
import { ExcelData, SortConfig } from '../types';

interface DataTableProps {
  data: ExcelData | null;
}

const DataTable: React.FC<DataTableProps> = ({ data }) => {
  const [sortConfig, setSortConfig] = useState<SortConfig | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredData, setFilteredData] = useState<any[][]>([]);

  useEffect(() => {
    if (!data) {
      setFilteredData([]);
      return;
    }

    let result = [...data.data];

    // Apply search filter
    if (searchTerm) {
      result = result.filter(row => 
        row.some((cell: any) => 
          String(cell).toLowerCase().includes(searchTerm.toLowerCase())
        )
      );
    }

    // Apply sorting
    if (sortConfig) {
      const columnIndex = data.headers.findIndex(header => header === sortConfig.key);
      
      if (columnIndex !== -1) {
        result.sort((a, b) => {
          const valueA = a[columnIndex];
          const valueB = b[columnIndex];
          
          // Handle different data types
          if (typeof valueA === 'number' && typeof valueB === 'number') {
            return sortConfig.direction === 'asc' ? valueA - valueB : valueB - valueA;
          }
          
          // Default to string comparison
          const strA = String(valueA || '').toLowerCase();
          const strB = String(valueB || '').toLowerCase();
          
          if (strA < strB) return sortConfig.direction === 'asc' ? -1 : 1;
          if (strA > strB) return sortConfig.direction === 'asc' ? 1 : -1;
          return 0;
        });
      }
    }

    setFilteredData(result);
  }, [data, searchTerm, sortConfig]);

  const requestSort = (key: string) => {
    let direction: 'asc' | 'desc' = 'asc';
    
    if (sortConfig && sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    
    setSortConfig({ key, direction });
  };

  if (!data) {
    return (
      <div className="text-center p-8 text-gray-500">
        尚未上傳檔案或無資料可顯示
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold">Excel 資料表</h2>
        <div className="relative">
          <input
            type="text"
            placeholder="搜尋..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 pr-4 py-2 border rounded-lg"
          />
          <Search className="absolute left-3 top-2.5 text-gray-400" size={18} />
        </div>
      </div>
      
      <div className="overflow-x-auto">
        <table className="min-w-full bg-white border border-gray-200">
          <thead>
            <tr className="bg-gray-100">
              {data.headers.map((header, index) => (
                <th 
                  key={index}
                  className="px-4 py-2 text-left border-b cursor-pointer hover:bg-gray-200"
                  onClick={() => requestSort(header)}
                >
                  <div className="flex items-center">
                    <span>{header}</span>
                    <ArrowUpDown size={14} className="ml-1 text-gray-500" />
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filteredData.length > 0 ? (
              filteredData.map((row, rowIndex) => (
                <tr key={rowIndex} className="hover:bg-gray-50">
                  {row.map((cell, cellIndex) => (
                    <td key={cellIndex} className="px-4 py-2 border-b">
                      {cell}
                    </td>
                  ))}
                </tr>
              ))
            ) : (
              <tr>
                <td 
                  colSpan={data.headers.length} 
                  className="px-4 py-8 text-center text-gray-500"
                >
                  無符合的資料
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      
      <div className="text-sm text-gray-500">
        顯示 {filteredData.length} 筆資料 (共 {data.data.length} 筆)
      </div>
    </div>
  );
};

export default DataTable;