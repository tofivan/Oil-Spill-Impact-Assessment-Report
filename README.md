# Oil-Spill-Impact-Assessment-Report
This script analyzes oil spill data from a NetCDF file simulated using OpenDrift, visualizes the spread of oil particles over time on a map, generates comprehensive reports in text and PDF formats, and creates animations (GIF and MP4) to illustrate the oil spill progression.

Overview

This Python program is designed to analyze oil spill data from a NetCDF (.nc) file, visualize the spread of oil particles over time on a map, generate reports in both text and PDF formats, and create animations (GIF and MP4) showcasing the oil spill's progression. The program leverages several libraries, including xarray for data handling, matplotlib and cartopy for visualization, reportlab for PDF generation, and imageio and ffmpeg for creating animations.

Key Components
Imports

Data Handling: xarray, numpy, pandas
Date and Time: datetime, timedelta
Visualization: matplotlib, cartopy
Mapping: Custom tile providers (ESRI_Imagery, ESRI_Maps, Google_Maps) extending cartopy's OSM class
PDF Generation: reportlab
Animation: imageio, ffmpeg (via subprocess)
Utilities: os, textwrap, subprocess
Font Setup

set_chinese_font() dynamically detects and sets a Chinese font to render Chinese characters in plots and PDFs properly.
Custom Map Tile Providers

ESRI_Imagery and ESRI_Maps: Provide ESRI map tiles for different styles (streets, satellite, hybrid).
Google_Maps: Provides Google Maps tiles with customizable styles, language, and resolution.
MapTileProvider: Manages different tile providers based on the selected map style.
MapStyle Class

Configures the map's appearance, including resolution, style, zoom level, and coastline features.
Handles gridline calculations and map extent based on data.
Adds environmental features like land, coastline, and ocean-based on settings.
Report Class

Initialization: Sets up locations of interest, map style, and animation settings.
Data Analysis (analyze_nc_file):
Opens the NetCDF file and extracts particle data (status, longitude, latitude).
Identifies stranded particles and active particles.
Collects trajectories of particles.
Processes environmental data (wind speed, wave height).
Map Generation (visualize_particles_at_time):
Plots particle positions on a map for a specific time index.
Draws historical trajectories of particles.
Marks key locations and adds legends.
Report Generation (generate, _generate_report_content, _format_report):
Compiles statistical data and map images into a structured report.
Assesses risk levels for predefined locations based on particle proximity.
Provides response suggestions based on risk assessments.
PDF Creation (generate_pdf):
Formats the report content and map images into a PDF document.
Handles Chinese text wrapping and font embedding.
Animation (generate_hourly_maps, create_gif):
Creates hourly map images and compiles them into a GIF animation.
Converts image sequences to an MP4 video using ffmpeg.
Animation Helper Function

create_mp4_from_images: Uses ffmpeg to convert a sequence of PNG images into an MP4 video, ensuring correct frame order and encoding.
Main Function (main)

Setup Paths: Defines directories and file paths for inputs and outputs.
Map Style Configuration: Initializes map styling parameters.
Report Generation Workflow:
Analyze the NetCDF file to extract and process data.
Generates hourly map images for animation.
Creates GIF and MP4 animations from the map images.
Generates a textual and PDF report summarizing the impact of the oil spill.
Error Handling: Catches and reports any exceptions during execution.
Execution Entry Point

The main() function is called when the script is executed directly, initiating the entire workflow.
Usage Workflow
Data Preparation: Ensure that the NetCDF file (combined_output.nc) containing oil spill data is in the specified base_dir.
Execution: Run the script. It will:
Analyze the data to identify active and stranded oil particles.
Generate hourly maps showing the spread of oil particles.
Compile these maps into GIF and MP4 animations.
Create a detailed report in both text and PDF formats, including risk assessments for predefined locations.
Outputs: The program will produce:
A text report (oil_impact_report.txt)
A PDF report (oil_impact_report.pdf)
Animated GIF (oil_spill_animation.gif) and MP4 (oil_spill_animation.mp4) files
Hourly map images stored in the animation directory
Important Considerations
Dependencies: Ensure all required libraries are installed (xarray, numpy, pandas, matplotlib, cartopy, reportlab, imageio, ffmpeg).
Font Availability: The script attempts to find and register a Chinese font for proper rendering. Ensure that the necessary font files are available on the system.
ffmpeg Installation: ffmpeg must be installed and accessible via the system's PATH for MP4 generation.
Data Validity: The script includes extensive error checking and logging to gracefully handle invalid or out-of-range data points.



中文說明:

概述
這個Python程式旨在分析來自OpenDrift油擴散模擬的NetCDF（.nc）檔案，並在地圖上可視化油粒子的擴散過程。它還能生成包含文字和PDF格式的中文報告，並製作展示油污染擴散動態的動畫（GIF和MP4）。程式使用了多個庫，包括xarray處理數據，matplotlib和cartopy進行視覺化，reportlab生成PDF，以及imageio和ffmpeg製作動畫。

主要組件
導入庫

數據處理: xarray, numpy, pandas
日期和時間: datetime, timedelta
視覺化: matplotlib, cartopy
地圖: 自定義圖磚提供者（ESRI_Imagery, ESRI_Maps, Google_Maps）擴展了cartopy的OSM類
PDF生成: reportlab
動畫: imageio, ffmpeg（透過subprocess）
工具: os, textwrap, subprocess
字體設置

set_chinese_font()動態檢測並設置中文字體，以確保在繪圖和PDF中正確顯示中文字符。
自定義地圖圖磚提供者

ESRI_Imagery和ESRI_Maps: 提供ESRI地圖圖磚，支持不同樣式（街道、衛星、混合）。
Google_Maps: 提供Google地圖圖磚，支持自定義樣式、語言和解析度。
MapTileProvider: 根據選擇的地圖樣式管理不同的圖磚提供者。
MapStyle類

配置地圖的外觀，包括解析度、樣式、縮放級別和海岸線特徵。
根據數據計算網格線和地圖範圍。
根據設置添加環境特徵，如陸地、海岸線和海洋。
Report類

初始化: 設定關注地點、地圖樣式和動畫設置。
數據分析（analyze_nc_file）:
打開NetCDF檔案，提取粒子數據（狀態、經度、緯度）。
識別擱淺粒子和活動粒子。
收集粒子的軌跡。
處理環境數據（風速、波高）。
地圖生成（visualize_particles_at_time）:
為特定時間索引在地圖上繪製粒子位置。
繪製粒子的歷史軌跡。
標註關鍵地點並添加圖例。
報告生成（generate, _generate_report_content, _format_report）:
將統計數據和地圖圖像編輯成結構化報告。
根據粒子與預定地點的距離評估風險等級。
根據風險評估提供應變建議。
PDF創建（generate_pdf）:
將報告內容和地圖圖像格式化為PDF文檔。
處理中文文本自動換行和字體嵌入。
動畫（generate_hourly_maps, create_gif）:
創建每小時的地圖圖像，並將其編輯成GIF動畫。
使用ffmpeg將圖像序列轉換為MP4視頻。
動畫輔助函數

create_mp4_from_images: 使用ffmpeg將一系列PNG圖像轉換為MP4視頻，確保幀順序和編碼正確。
主函數（main）

設置路徑: 定義輸入和輸出的目錄及文件路徑。
地圖樣式配置: 初始化地圖樣式參數。
報告生成流程:
分析NetCDF檔案以提取和處理資料。
生成每小時的地圖圖像以製作動畫。
將地圖圖像編輯成GIF和MP4動畫。
生成包含文字和PDF格式的詳細報告，並進行風險評估。
錯誤處理: 捕捉並報告執行過程中的任何異常。
執行入口

當腳本被直接執行時，會調用main()函數，啟動整個工作流程。
使用流程
數據準備: 確保NetCDF檔案（combined_output.nc）包含油污染資料，並位於指定的base_dir目錄中。
執行程式: 運行腳本，程式將：
分析數據以識別活動粒子和擱淺粒子。
生成每小時的地圖影像，展示油污染擴散範圍。
將地圖影像儲存成GIF和MP4動畫。
創建包含統計數據和風險評估的中文文字和PDF報告。
輸出結果: 程式將產生：
文字報告（oil_impact_report.txt）
PDF報告（oil_impact_report.pdf）
動畫GIF（oil_spill_animation.gif）和MP4（oil_spill_animation.mp4）
每小時的地圖圖像，存儲在animation目錄中
重要考量
依賴性: 確保安裝所有必需的庫（xarray, numpy, pandas, matplotlib, cartopy, reportlab, imageio, ffmpeg）。
字體可用性: 程式嘗試尋找並註冊中文字體以正確使用中文字符。確保系統中有必要的字體文件。
ffmpeg安裝: 需要安裝ffmpeg並確保其在系統的PATH中，以便生成MP4視頻。
資料有效性: 程式包含廣泛的錯誤檢查和日誌記錄，以優雅地處理無效或超出範圍的資料。
