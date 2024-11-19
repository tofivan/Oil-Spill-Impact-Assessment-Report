# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:02:41 2024

@author: liouyufu

=======================================
Oil Spill Impact Assessment Report
=======================================

Author: [liouyufu]
Date: 2024-11-19
Version: 0.8

Description:
This script analyzes oil spill data from a NetCDF file simulated using OpenDrift, visualizes the spread of oil particles over time on a map,
generates comprehensive reports in text and PDF formats, and creates animations (GIF and MP4) to illustrate the oil spill progression.

The analysis includes identifying active and stranded oil particles, processing environmental data such as wind speed and wave height,
and assessing risk levels for predefined locations based on particle proximity. The visualizations leverage libraries like matplotlib and cartopy,
while PDF reports are generated using reportlab. Animations are created using imageio and ffmpeg.

License:
© 2024 [liouyufu]. All rights reserved.
[ MIT License].

Contact:
[liouyfu@g.ncu.edu.tw]

"""

import xarray as xr
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Literal
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import cartopy.io.img_tiles as cimgt
import matplotlib
import matplotlib.font_manager as fm
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
import textwrap
import pandas as pd
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy
from cartopy.io.img_tiles import OSM
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import subprocess
import imageio.v2 as imageio
from typing import List



print(cartopy.__version__)

def set_chinese_font():
    """動態檢測和設置中文字體"""
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'AR PL UMing CN']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            matplotlib.rc('font', family=font_name)
            print(f"使用字體: {font_name}")
            return
    print("未找到中文字體！可能無法正確顯示中文。")
    
class ESRI_Imagery(OSM):
    """自定義的 ESRI 影像瓦片提供者"""
    def _image_url(self, tile):
        x, y, z = tile
        return ('https://server.arcgisonline.com/ArcGIS/rest/services/'
                'World_Imagery/MapServer/tile/{z}/{y}/{x}'.format(z=z, y=y, x=x))
    
class ESRI_Maps(OSM):
    """ESRI 地圖服務提供者"""
    def __init__(self, style: Literal['streets', 'satellite', 'hybrid'] = 'satellite'):
        self.style = style
        super().__init__()
        
    def _image_url(self, tile):
        x, y, z = tile
        service_url = {
            'streets': 'World_Street_Map',
            'satellite': 'World_Imagery',
            'hybrid': 'World_Imagery/MapServer'
        }.get(self.style)
        
        if self.style == 'hybrid':
            return (
                'https://services.arcgisonline.com/ArcGIS/rest/services/' +
                f'{service_url}/tile/{z}/{y}/{x}'
            )
        return (
            'https://server.arcgisonline.com/ArcGIS/rest/services/' +
            f'{service_url}/MapServer/tile/{z}/{y}/{x}'
        )

class Google_Maps(OSM):
    """Google Maps 影像提供者類"""
    def __init__(self, 
                 style: Literal['roadmap', 'satellite', 'terrain', 'hybrid'] = 'satellite',
                 language: str = 'zh-TW',  # 設置語言
                 scale: int = 2):          # 控制解析度倍數
        self.style = style
        self.language = language
        self.scale = scale  # 1 為標準解析度，2 為高解析度
        super().__init__()
        
    def _image_url(self, tile):
        x, y, z = tile
        style_dict = {
            'roadmap': 'm',
            'satellite': 's',
            'terrain': 't',
            'hybrid': 'y'
        }
        map_type = style_dict.get(self.style, 's')
        return (
            'http://mt0.google.com/vt/lyrs=' + 
            f'{map_type}&hl={self.language}&x={x}&y={y}&z={z}&s=Ga' +
            f'&scale={self.scale}'  # 添加 scale 參數
        )
    
class MapTileProvider:
    """地圖圖磚提供者類"""
    def __init__(self, style: str = 'terrain'):
        self.style = style
        self._init_tile_source()

    def _init_tile_source(self):
        """初始化圖磚來源"""
        # Google Maps 相關
        if self.style == 'google_map':
            self.tile_source = Google_Maps('roadmap', scale=2)
        elif self.style == 'google_satellite':
            self.tile_source = Google_Maps('satellite', scale=2)
        elif self.style == 'google_hybrid':
            self.tile_source = Google_Maps('hybrid', scale=2)
        # ESRI 相關
        elif self.style == 'esri_street':
            self.tile_source = ESRI_Maps('streets')
        elif self.style == 'esri_satellite':
            self.tile_source = ESRI_Maps('satellite')
        elif self.style == 'esri_hybrid':
            self.tile_source = ESRI_Maps('hybrid')
        # OpenStreetMap 相關
        elif self.style == 'osm_street':
            self.tile_source = cimgt.OSM()
        elif self.style == 'osm_terrain':
            self.tile_source = cimgt.Stamen('terrain-background')
        elif self.style == 'osm_toner':
            self.tile_source = cimgt.Stamen('toner')
        # 其他
        elif self.style == 'satellite':
            self.tile_source = ESRI_Imagery()
        elif self.style == 'street':
            self.tile_source = cimgt.OSM()
        else:  # terrain
            self.tile_source = cimgt.Stamen('terrain')
            
class MapStyle:
    """地圖樣式設定類"""
    def __init__(self, 
                 use_high_res: bool = False, 
                 map_style: Literal['terrain', 'satellite', 'street', 'google_map', 
                                  'google_satellite', 'google_hybrid', 
                                  'esri_street', 'esri_satellite', 'esri_hybrid',
                                  'osm_street', 'osm_terrain', 'osm_toner'] = 'terrain',
                 tile_zoom: int = 8,
                 show_gshhg: bool = False,  #  GSHHG 海岸線可以開啟True或關閉False
                 gshhg_resolution: Literal['c', 'l', 'i', 'h', 'f'] = 'f'):  # 新增解析度選項
        self.use_high_res = use_high_res
        self.map_style = map_style
        self.tile_zoom = tile_zoom
        self.show_gshhg = show_gshhg
        self.gshhg_resolution = gshhg_resolution
        self.tile_provider = MapTileProvider(map_style)
        
        # 初始化 GSHHG 特徵
        self.gshhg_land = cfeature.GSHHSFeature(
            scale=gshhg_resolution,
            levels=[1],  # 1 表示陸地
            edgecolor='black',
            facecolor='none',
            linewidth=0.5
        )
        
        self.gshhg_coastline = cfeature.GSHHSFeature(
            scale=gshhg_resolution,
            levels=[1],
            edgecolor='black',
            facecolor='none',
            linewidth=0.8
        )
        
        # 保留原有的 Natural Earth 特徵，以備不時之需
        self.coastline_10m = cfeature.NaturalEarthFeature(
            'physical', 'coastline', '10m',
            edgecolor='black', facecolor='none', linewidth=0.8
        )
        
        self.land_10m = cfeature.NaturalEarthFeature(
            'physical', 'land', '10m',
            edgecolor='face', facecolor='lightgray'
        )


    
    def setup_map_style(self, ax: plt.Axes, extent: List[float]) -> plt.Axes:
        """設置地圖基本樣式"""
        try:
            # 設置地圖範圍
            ax.set_extent(extent)
            
            # 計算經緯度範圍
            lon_range = extent[1] - extent[0]
            lat_range = extent[3] - extent[2]
            
            # 計算中心緯度
            center_lat = (extent[2] + extent[3]) / 2
            
            # 根據中心緯度計算經度間距修正係數
            # 由於經度間距會隨緯度變化，需要進行修正以保持正方形
            lon_correction = np.cos(np.radians(center_lat))
            
            # 計算網格間距
            # 使用緯度間距作為基準，經度間距根據投影關係調整
            grid_size = round(min(lon_range, lat_range) / 5, 1)  # 取較小範圍除以5作為基準
            lat_step = grid_size
            lon_step = grid_size / lon_correction  # 修正經度間距以保持正方形
            
            # 確保間隔不會太小或太大
            lat_step = max(min(lat_step, 1.0), 0.1)
            lon_step = max(min(lon_step, 1.0), 0.1)
            
            # 如果不是地形圖，先添加底圖
            if self.map_style != 'terrain':
                ax.add_image(self.tile_provider.tile_source, self.tile_zoom, alpha=1)
            
            # 根據設定決定是否顯示 GSHHG 海岸線
            if self.show_gshhg:
                if self.map_style == 'terrain':
                    ax.add_feature(self.gshhg_land, zorder=1)
                ax.add_feature(self.gshhg_coastline, zorder=5)
            else:
                if self.use_high_res:
                    if self.map_style == 'terrain':
                        ax.add_feature(self.land_10m, zorder=1)
                        ax.add_feature(self.coastline_10m, zorder=2)
                else:
                    if self.map_style == 'terrain':
                        ax.add_feature(cfeature.LAND, color='lightgray', zorder=1)
                        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=2)
            
            # 計算網格線位置
            # 確保網格線從整數倍的間距開始
            lon_start = np.ceil(extent[0] / lon_step) * lon_step
            lon_end = np.floor(extent[1] / lon_step) * lon_step
            lat_start = np.ceil(extent[2] / lat_step) * lat_step
            lat_end = np.floor(extent[3] / lat_step) * lat_step
            
            # 添加網格線
            gl = ax.gridlines(
                draw_labels=True,
                linewidth=0.8,  # 調整線條寬度
                color='gray',
                linestyle='--',  # 使用虛線
                alpha=0.7,  # 增加透明度
                xlocs=np.arange(lon_start, lon_end + lon_step, lon_step),
                ylocs=np.arange(lat_start, lat_end + lat_step, lat_step)
            )
            
            # 設置網格標籤樣式
            gl.top_labels = False  # 隱藏頂部標籤
            gl.right_labels = True  # 顯示右側標籤
            gl.xlabel_style = {'size': 10, 'weight': 'bold'}  # 加粗字體
            gl.ylabel_style = {'size': 10, 'weight': 'bold'}
            gl.xformatter = matplotlib.ticker.FormatStrFormatter('%.2f°')  # 經度顯示到小數點2位
            gl.yformatter = matplotlib.ticker.FormatStrFormatter('%.2f°')  # 緯度顯示到小數點2位
    
            # 加粗圖框
            ax.spines['geo'].set_linewidth(1.5)  # 設置圖框加粗
            
            # 針對不同地圖樣式添加海洋
            if self.map_style in ['terrain', 'street', 'google_map'] and not self.show_gshhg:
                ax.add_feature(
                    cfeature.OCEAN.with_scale('50m'),
                    facecolor='lightblue',
                    alpha=1
                )
            
            # 輸出網格資訊用於調試
            print(f"\n=== 網格設置資訊 ===")
            print(f"中心緯度: {center_lat:.2f}°")
            print(f"經度修正係數: {lon_correction:.3f}")
            print(f"網格間距: {grid_size:.2f}°")
            print(f"實際經度間距: {lon_step:.2f}°")
            print(f"實際緯度間距: {lat_step:.2f}°")
            
        except Exception as e:
            print(f"設置地圖樣式時發生錯誤: {str(e)}")
            raise
        
        return ax
  
class Report:
    def __init__(self, 
                 map_style: Optional[MapStyle] = None,
                 show_location_labels: bool = False,  # 控制地標文字顯示False
                 show_location_markers: bool = False,  # 控制地標標記顯示False
                 location_label_offset: float = 0.01):  # 文字框偏移量
        # 原有的初始化保持不變
        self.locations = {
            '台北港、淡水河口': (121.380370, 25.186088),
            '林口發電廠': (121.291961, 25.138925),
            '大潭三接港': (121.032961, 25.061093),
            '新北市三芝海域': (121.442245, 25.247011)
        }
        self.map_style = map_style or MapStyle()
        self.fixed_start_time = None
        self.fixed_end_time = None
        set_chinese_font()
        
        # 新增顯示控制參數
        self.show_location_labels = show_location_labels
        self.show_location_markers = show_location_markers
        self.location_label_offset = location_label_offset
        
        # 新增動畫相關的設定
        self.animation_dpi = 150  # GIF的解析度
        self.animation_interval = 500  # 幀間隔(毫秒)

    def _calculate_map_extent(self, ds: xr.Dataset) -> List[float]:
        """計算基於所有時間步驟的粒子活動最大範圍"""
        try:
            # 獲取所有時間步驟的經緯度值
            lon_vals = ds.lon.values
            lat_vals = ds.lat.values
            
            # 設定合理的座標範圍限制
            MAX_LON = 180
            MIN_LON = -180
            MAX_LAT = 90
            MIN_LAT = -90
            
            # 過濾有效值及合理範圍
            valid_mask = (
                np.isfinite(lon_vals) & 
                np.isfinite(lat_vals) &
                (lon_vals > MIN_LON) & (lon_vals < MAX_LON) &
                (lat_vals > MIN_LAT) & (lat_vals < MAX_LAT)
            )
            
            # 獲取所有有效的經緯度值
            valid_lon = lon_vals[valid_mask]
            valid_lat = lat_vals[valid_mask]
            
            if len(valid_lon) == 0 or len(valid_lat) == 0:
                raise ValueError("找不到有效的經緯度數據")
            
            # 計算整體邊界
            lon_min = np.nanmin(valid_lon)
            lon_max = np.nanmax(valid_lon)
            lat_min = np.nanmin(valid_lat)
            lat_max = np.nanmax(valid_lat)
            
            print(f"\n=== 原始數據統計 ===")
            print(f"所有經度值範圍: {np.min(lon_vals):.3f}°E - {np.max(lon_vals):.3f}°E")
            print(f"所有緯度值範圍: {np.min(lat_vals):.3f}°N - {np.max(lat_vals):.3f}°N")
            print(f"有效數據點數量: {np.sum(valid_mask)}")
            print(f"異常數據點數量: {np.sum(~valid_mask)}")
            
            # 添加邊距（可以調整百分比）
            margin_percent = 0.1  # 10% 邊距
            lon_margin = (lon_max - lon_min) * margin_percent
            lat_margin = (lat_max - lat_min) * margin_percent
            
            extent = [
                float(lon_min - lon_margin),
                float(lon_max + lon_margin),
                float(lat_min - lat_margin),
                float(lat_max + lat_margin)
            ]
            
            print(f"\n=== 地圖範圍計算結果 ===")
            print(f"有效經度範圍: {lon_min:.3f}°E - {lon_max:.3f}°E")
            print(f"有效緯度範圍: {lat_min:.3f}°N - {lat_max:.3f}°N")
            print(f"添加 {margin_percent*100}% 邊距")
            print(f"最終範圍: {extent}")
            
            return extent
                
        except Exception as e:
            print(f"計算地圖範圍時發生錯誤: {str(e)}")
            raise


    def generate_hourly_maps(self, ds: xr.Dataset, stats: Dict, 
                           trajectories: List, base_output: str) -> List[str]:
        """生成每小時的模擬圖"""
        hourly_files = []
        time_values = pd.to_datetime(ds.time.values)
        total_hours = int((time_values[-1] - time_values[0]).total_seconds() / 3600)

        print(f"\n開始生成每小時模擬圖，總計 {total_hours+1} 張...")
        
        for hour in range(total_hours + 1):
            # 找到最接近目標時間的時間索引
            target_time = time_values[0] + pd.Timedelta(hours=hour)
            time_idx = np.abs(time_values - target_time).argmin()
            
            # 生成輸出檔案名稱
            output_file = os.path.splitext(base_output)[0] + f"_h{hour:03d}.png"
            print(f"生成第 {hour} 小時模擬圖: {os.path.basename(output_file)}")
            
            # 繪製該時間點的模擬圖
            self.visualize_particles_at_time(
                ds, time_idx, stats, trajectories, 
                output_file, float(hour),
                is_animation_frame=True  # 新增參數表示這是動畫幀
            )
            
            hourly_files.append(output_file)
        
        return hourly_files
    
    
    def create_gif(self, image_files: List[str], output_gif: str) -> None:
        """將多張圖片合成為GIF動畫"""
        try:
            print(f"\n開始生成GIF動畫: {output_gif}")
            
            # 讀取所有圖片
            images = []
            for filename in image_files:
                images.append(imageio.imread(filename))
            
            # 生成GIF
            imageio.mimsave(
                output_gif, 
                images, 
                duration=self.animation_interval/1000.0,  # 轉換為秒
                loop=0  # 永續循環
            )
            
            print(f"GIF動畫已生成: {output_gif}")
            print(f"- 總幀數: {len(images)}")
            print(f"- 幀間隔: {self.animation_interval}ms")
            
        except Exception as e:
            print(f"生成GIF時發生錯誤: {str(e)}")
            raise

    def analyze_nc_file(self, nc_file: str) -> Tuple[Dict, np.ndarray, np.ndarray, List, xr.Dataset]:
        """分析nc檔案中的油污染資訊"""
        try:
            ds = xr.open_dataset(nc_file)
            
            print("\n=== 檔案資訊診斷 ===")
            print(f"時間維度大小: {len(ds.time)}")
            print(f"粒子數量: {ds.status.shape[0]}")
            
            # 獲取數據
            status = ds.status.values
            lon = ds.lon.values
            lat = ds.lat.values
            
            # 儲存固定的時間範圍
            self.fixed_start_time = pd.to_datetime(ds.time.values[0])
            self.fixed_end_time = pd.to_datetime(ds.time.values[-1])
            
            # 對擱淺粒子，找到最後的有效位置
            final_status = status[:, -1]
            stranded_indices = np.where(final_status == -2147483647)[0]
            
            print("\n=== Status 變量資訊 ===")
            print("Status attributes:", ds.status.attrs)
            unique_status = np.unique(final_status)
            print("所有可能的status值:", unique_status)
            for s in unique_status:
                count = np.sum(final_status == s)
                print(f"Status {s}: {count} 個粒子")
            
            # 找到擱淺粒子最後的有效位置
            stranded_lons = []
            stranded_lats = []
            for idx in stranded_indices:
                for t in range(lon.shape[1]-1, -1, -1):
                    if (np.isfinite(lon[idx, t]) and np.isfinite(lat[idx, t]) and
                        110 < lon[idx, t] < 130 and 15 < lat[idx, t] < 35):  # 擴大有效範圍
                        stranded_lons.append(lon[idx, t])
                        stranded_lats.append(lat[idx, t])
                        break
    
            # 活動粒子使用最後時刻的位置
            active_mask = final_status == 0
            final_lon = lon[:, -1]
            final_lat = lat[:, -1]
            
            valid_active = (
                active_mask & 
                np.isfinite(final_lon) & 
                np.isfinite(final_lat) & 
                (final_lon > 110) & (final_lon < 130) &  # 擴大有效範圍
                (final_lat > 15) & (final_lat < 35)      # 擴大有效範圍
            )
            
            active_points = np.column_stack([
                final_lon[valid_active],
                final_lat[valid_active]
            ])
            
            stranded_points = np.array(list(zip(stranded_lons, stranded_lats))) if stranded_lons else np.empty((0, 2))
    
            print("\n=== 粒子統計 ===")
            print(f"活動粒子: {len(active_points)}")
            print(f"擱淺粒子: {len(stranded_points)}")
            
            # 收集軌跡
            trajectories = []
            for i in range(status.shape[0]):
                particle_traj = []
                for t in range(status.shape[1]):
                    curr_lon = lon[i, t]
                    curr_lat = lat[i, t]
                    if (np.isfinite(curr_lon) and np.isfinite(curr_lat) and
                        110 < curr_lon < 130 and 15 < curr_lat < 35):  # 擴大有效範圍
                        particle_traj.append((curr_lon, curr_lat))
                    if status[i, t] == -2147483647:
                        break
                if particle_traj:
                    trajectories.append(particle_traj)
    
            # 處理環境數據和統計資訊
            stats = self._process_environmental_data(ds, active_points, stranded_points)
            
            return stats, active_points, stranded_points, trajectories, ds
            
        except Exception as e:
            print(f"分析NC檔案時發生錯誤: {str(e)}")
            raise
            
    def _process_environmental_data(self, ds: xr.Dataset, 
                                  active_points: np.ndarray, 
                                  stranded_points: np.ndarray) -> Dict:
        """處理環境數據並生成統計資訊"""
        # 處理風速數據
        x_wind = ds.x_wind.values if 'x_wind' in ds else np.zeros_like(ds.time)
        y_wind = ds.y_wind.values if 'y_wind' in ds else np.zeros_like(ds.time)
        
        valid_x_wind = np.where((np.isfinite(x_wind)) & (np.abs(x_wind) < 100), x_wind, 0)
        valid_y_wind = np.where((np.isfinite(y_wind)) & (np.abs(y_wind) < 100), y_wind, 0)
        
        wind_speed = np.sqrt(np.clip(valid_x_wind, -100, 100)**2 + 
                           np.clip(valid_y_wind, -100, 100)**2)
        valid_wind = wind_speed[np.isfinite(wind_speed) & (wind_speed < 100)]
        
        # 處理波高數據
        if 'sea_surface_wave_significant_height' in ds:
            wave_height = ds.sea_surface_wave_significant_height.values
            valid_wave = wave_height[np.isfinite(wave_height) & (wave_height < 100)]
        else:
            valid_wave = np.array([0])
        
        # 計算位置範圍
        all_points = []
        if len(active_points) > 0:
            all_points.extend(active_points)
        if len(stranded_points) > 0:
            all_points.extend(stranded_points)
            
        if all_points:
            all_points = np.array(all_points)
            lon_range = (float(np.min(all_points[:, 0])), float(np.max(all_points[:, 0])))
            lat_range = (float(np.min(all_points[:, 1])), float(np.max(all_points[:, 1])))
        else:
            lon_range = (float(ds.attrs.get('seed_lon', 121.0)), float(ds.attrs.get('seed_lon', 121.0)))
            lat_range = (float(ds.attrs.get('seed_lat', 25.0)), float(ds.attrs.get('seed_lat', 25.0)))
        
        # 生成統計資訊
        stats = {
            'start_time': pd.to_datetime(ds.time.values[0]),
            'end_time': pd.to_datetime(ds.time.values[-1]),
            'start_point': (float(ds.attrs.get('seed_lon', 121.0)), float(ds.attrs.get('seed_lat', 25.0))),
            'oil_type': ds.attrs.get('seed_oiltype', ds.attrs.get('config_seed:oil_type', 'Unknown')),
            'time_steps': len(ds.time),
            'lon_range': lon_range,
            'lat_range': lat_range,
            'active_particles': len(active_points),
            'stranded_particles': len(stranded_points),
            'wind_speed_range': (float(np.min(valid_wind)), float(np.max(valid_wind))),
            'wave_height_range': (float(np.min(valid_wave)), float(np.max(valid_wave)))
        }
        
        return stats
    
    def _get_title(self, current_time: pd.Timestamp, hours: float) -> str:
        """生成圖表標題"""
        # 使用儲存的固定時間範圍
        fixed_start_utc8 = self.fixed_start_time.tz_localize('UTC').tz_convert('Asia/Taipei')
        fixed_end_utc8 = self.fixed_end_time.tz_localize('UTC').tz_convert('Asia/Taipei')
        fixed_time_str = f"開始：{fixed_start_utc8.strftime('%Y/%m/%d %H:%M')} - 結束：{fixed_end_utc8.strftime('%Y/%m/%d %H:%M')} (UTC+8)"
        
        # 當前模擬時間
        current_time_utc8 = current_time.tz_localize('UTC').tz_convert('Asia/Taipei')
        current_time_str = f"{current_time_utc8.strftime('%Y-%m-%d %H:%M:%S')} (UTC+8)"
        
        if hours == 0:
            title = (f"油污染擴散分布圖\n"
                    f"{fixed_time_str}\n"
                    f"{current_time_str}")
        else:
            title = (f"油污染擴散分布圖 ({hours:.1f}小時後)\n"
                    f"{fixed_time_str}\n"
                    f"{current_time_str}")
        
        return title
    
    def _plot_particles(self, ax: plt.Axes, active_points: np.ndarray, 
                       stranded_points: np.ndarray, stats: Dict) -> None:
        """繪製粒子分布"""
        if len(active_points) > 0:
            ax.scatter(active_points[:, 0], active_points[:, 1],
                      s=20, c='blue', alpha=0.6, zorder=3,
                      label=f'海面漂移粒子 ({len(active_points)}個)')
        
        if len(stranded_points) > 0:
            ax.scatter(stranded_points[:, 0], stranded_points[:, 1],
                      s=15, c='black', alpha=0.8, zorder=3,
                      label=f'岸際擱淺粒子 ({len(stranded_points)}個)')
        
        # 繪製起點
        ax.plot(stats['start_point'][0], stats['start_point'][1], 'r*',
                markersize=15, label='溢油位置', zorder=4)
        
        # 標註重要地點（根據顯示設置）
        if self.show_location_labels or self.show_location_markers:
            for name, (lon, lat) in self.locations.items():
                # 繪製標記（三角形）
                if self.show_location_markers:
                    ax.plot(lon, lat, 'k^', markersize=8, zorder=5)
                
                # 繪製文字標籤
                if self.show_location_labels:
                    ax.text(lon, lat + self.location_label_offset, name,
                           fontsize=10, ha='center', va='bottom',
                           bbox=dict(facecolor='white', 
                                   alpha=0.8, 
                                   edgecolor='black', 
                                   pad=2),
                           zorder=4)
        
        # 添加圖例
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1.0),
                 fontsize=9, frameon=True, facecolor='white', edgecolor='none')
        
    
    def _get_stranded_points(self, ds: xr.Dataset, time_idx: int) -> np.ndarray:
        """獲取擱淺粒子位置"""
        stranded_points = []
        stranded_count = 0
        filtered_out = 0
        invalid_coords = 0
        
        print(f"\n=== 擱淺粒子位置診斷（時間點：{time_idx}） ===")
        
        for particle_idx in range(ds.status.shape[0]):
            is_stranded = False
            stranded_position = None
            
            # 檢查當前粒子是否在此時間點之前或當前時間點擱淺
            for t in range(time_idx + 1):
                if ds.status.values[particle_idx, t] == -2147483647:
                    stranded_count += 1
                    is_stranded = True
                    
                    # 向前尋找擱淺前的最後有效位置
                    search_time = max(0, t-1)  # 使用擱淺前的時間點
                    lon_val = ds.lon.values[particle_idx, search_time]
                    lat_val = ds.lat.values[particle_idx, search_time]
                    
                    # 檢查座標有效性
                    if not np.isfinite(lon_val) or not np.isfinite(lat_val):
                        invalid_coords += 1
                        continue
                        
                    # 過濾明顯不合理的值和超出範圍的值
                    if (lon_val > 1000 or lat_val > 1000 or  # 過濾極大值
                        lon_val < 119 or lon_val > 122 or    # 調整為台灣周圍的合理範圍
                        lat_val < 23 or lat_val > 26):       # 調整為台灣周圍的合理範圍
                        filtered_out += 1
                        continue
                    
                    stranded_position = [lon_val, lat_val]
                    break
            
            if is_stranded and stranded_position is not None:
                stranded_points.append(stranded_position)
        
        print(f"總擱淺粒子數: {stranded_count}")
        print(f"成功找到位置的粒子數: {len(stranded_points)}")
        print(f"無效座標數: {invalid_coords}")
        print(f"超出範圍粒子數: {filtered_out}")
        print(f"篩選條件: 119°E-122°E, 23°N-26°N")  # 更新為實際使用的範圍
        
        if len(stranded_points) > 0:
            points_array = np.array(stranded_points)
            print(f"擱淺點經度範圍: {np.min(points_array[:,0]):.3f}°E - {np.max(points_array[:,0]):.3f}°E")
            print(f"擱淺點緯度範圍: {np.min(points_array[:,1]):.3f}°N - {np.max(points_array[:,1]):.3f}°N")
        
        return np.array(stranded_points) if stranded_points else np.empty((0, 2))
    
    '''
    def visualize_particles_at_time(self, ds: xr.Dataset, time_idx: int, 
                              stats: Dict, trajectories: List, 
                              output_file: str, hours: float,
                              is_animation_frame: bool = False) -> None:
        """在指定時間點繪製粒子分布圖"""
        try:
            print(f"\n=== 時間點 {time_idx} 的粒子狀態分析 ===")
            current_time = pd.to_datetime(ds.time.values[time_idx])
            print(f"當前時間: {current_time}")
            
            current_status = ds.status.values[:, time_idx]
            current_lon = ds.lon.values[:, time_idx]
            current_lat = ds.lat.values[:, time_idx]
            
            # 分析當前狀態
            unique_status = np.unique(current_status)
            for status in unique_status:
                count = np.sum(current_status == status)
                print(f"Status {status}: {count} 個粒子")
            
            # 過濾掉無效和不合理的座標
            valid_mask = (
                np.isfinite(current_lon) & 
                np.isfinite(current_lat)
            )
            
            # 分析座標有效性
            print(f"\n座標分析:")
            print(f"總粒子數: {len(current_lon)}")
            print(f"有效座標數: {np.sum(valid_mask)}")
            print(f"無效座標數: {len(current_lon) - np.sum(valid_mask)}")
            
            if np.sum(valid_mask) > 0:
                valid_lon = current_lon[valid_mask]
                valid_lat = current_lat[valid_mask]
                print(f"\n有效座標範圍:")
                print(f"經度範圍: {np.min(valid_lon):.3f}°E - {np.max(valid_lon):.3f}°E")
                print(f"緯度範圍: {np.min(valid_lat):.3f}°N - {np.max(valid_lat):.3f}°N")
            
            # 設置 dpi
            plt.rcParams['figure.dpi'] = 1000 if not is_animation_frame else self.animation_dpi
            
            # 計算地圖範圍
            map_extent = self._calculate_map_extent(ds)
            print(f"計算的地圖範圍: {map_extent}")
            
            # 根據範圍動態計算適合的圖片比例
            lon_range = map_extent[1] - map_extent[0]
            lat_range = map_extent[3] - map_extent[2]
            aspect_ratio = lon_range / lat_range
            
            # 根據範圍和是否為動畫幀來調整圖片尺寸
            if is_animation_frame:
                base_height = 6
                fig_width = base_height * aspect_ratio
                fig, ax = plt.subplots(
                    subplot_kw={'projection': ccrs.PlateCarree()},
                    figsize=(fig_width, base_height)
                )
            else:
                base_height = 8
                fig_width = base_height * aspect_ratio
                fig, ax = plt.subplots(
                    subplot_kw={'projection': ccrs.PlateCarree()},
                    figsize=(fig_width, base_height)
                )
                    
            ax = self.map_style.setup_map_style(ax, map_extent)
            
            # 過濾活動粒子
            active_mask = (
                (current_status == 0) & 
                valid_mask
            )
            active_points = np.column_stack([
                current_lon[active_mask], 
                current_lat[active_mask]
            ])
            
            # 獲取擱淺粒子
            stranded_points = self._get_stranded_points(ds, time_idx)
            
            # 繪製粒子
            self._plot_particles(ax, active_points, stranded_points, stats)
            
            # 設置標題
            current_time = pd.to_datetime(ds.time.values[time_idx])
            title = self._get_title(current_time, hours)
            plt.title(title, fontsize=12, pad=10)
            
            # 根據是否為動畫幀來調整保存參數
            if is_animation_frame:
                plt.savefig(output_file, 
                           bbox_inches='tight', 
                           dpi=self.animation_dpi,
                           pad_inches=0.1)
            else:
                plt.savefig(output_file, 
                       bbox_inches='tight',
                       dpi=1200,  # 提高輸出 DPI
                       pad_inches=0.1)
            
            plt.close()
            
        except Exception as e:
            print(f"繪製地圖時發生錯誤: {str(e)}")
            raise
     ''' 
    def visualize_particles_at_time(self, ds: xr.Dataset, time_idx: int, 
                              stats: Dict, trajectories: List, 
                              output_file: str, hours: float,
                              is_animation_frame: bool = False) -> None:
        """在指定時間點繪製粒子分布圖，包含完整軌跡"""
        try:
            print(f"\n=== 時間點 {time_idx} 的粒子狀態分析 ===")
            current_time = pd.to_datetime(ds.time.values[time_idx])
            print(f"當前時間: {current_time}")
            
            current_status = ds.status.values[:, time_idx]
            current_lon = ds.lon.values[:, time_idx]
            current_lat = ds.lat.values[:, time_idx]
            
            # 過濾掉無效和不合理的座標
            valid_mask = (
                np.isfinite(current_lon) & 
                np.isfinite(current_lat)
            )
            
            # 設置 dpi
            plt.rcParams['figure.dpi'] = 1000 if not is_animation_frame else self.animation_dpi
            
            # 計算地圖範圍
            map_extent = self._calculate_map_extent(ds)
            print(f"計算的地圖範圍: {map_extent}")
            
            # 根據範圍動態計算適合的圖片比例
            lon_range = map_extent[1] - map_extent[0]
            lat_range = map_extent[3] - map_extent[2]
            aspect_ratio = lon_range / lat_range
            
            # 根據範圍和是否為動畫幀來調整圖片尺寸
            if is_animation_frame:
                base_height = 6
                fig_width = base_height * aspect_ratio
                fig, ax = plt.subplots(
                    subplot_kw={'projection': ccrs.PlateCarree()},
                    figsize=(fig_width, base_height)
                )
            else:
                base_height = 8
                fig_width = base_height * aspect_ratio
                fig, ax = plt.subplots(
                    subplot_kw={'projection': ccrs.PlateCarree()},
                    figsize=(fig_width, base_height)
                )
                    
            ax = self.map_style.setup_map_style(ax, map_extent)
            
            # 繪製粒子的歷史軌跡
            print("繪製粒子軌跡...")
            for traj_idx in range(ds.lon.shape[0]):
                traj_lon = ds.lon.values[traj_idx, :time_idx+1]
                traj_lat = ds.lat.values[traj_idx, :time_idx+1]
                
                valid_traj_mask = np.isfinite(traj_lon) & np.isfinite(traj_lat)
                if np.any(valid_traj_mask):
                    ax.plot(traj_lon[valid_traj_mask], traj_lat[valid_traj_mask],
                            transform=ccrs.PlateCarree(), alpha=0.5, linewidth=0.3, color="gray")
            
            # 過濾活動粒子
            active_mask = (
                (current_status == 0) & 
                valid_mask
            )
            active_points = np.column_stack([
                current_lon[active_mask], 
                current_lat[active_mask]
            ])
            
            # 獲取擱淺粒子
            stranded_points = self._get_stranded_points(ds, time_idx)
            
            # 繪製粒子
            self._plot_particles(ax, active_points, stranded_points, stats)
            
            # 設置標題
            title = self._get_title(current_time, hours)
            plt.title(title, fontsize=12, pad=10)
            
            # 保存圖片
            plt.savefig(output_file, 
                        bbox_inches='tight', 
                        dpi=self.animation_dpi if is_animation_frame else 800,
                        pad_inches=0.1)
            plt.close()
            
        except Exception as e:
            print(f"繪製地圖時發生錯誤: {str(e)}")
            raise

        
    def generate(self, nc_file: str, locations: List[str], map_output: str) -> Tuple[str, List[Tuple[float, str]]]:
        """生成影響範圍報告和不同時間的地圖"""
        try:
            output_dir = os.path.dirname(map_output)
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"\n開始分析檔案: {nc_file}")
            stats, active_points, stranded_points, trajectories, ds = self.analyze_nc_file(nc_file)
            
            time_values = pd.to_datetime(ds.time.values)
            total_hours = (time_values[-1] - time_values[0]).total_seconds() / 3600
            
            target_hours = [12, 24, 48, 72]
            map_files = []
            
            for target_hour in target_hours:
                if target_hour > total_hours:
                    print(f"警告：{target_hour}小時超出模擬時間範圍（總時長：{total_hours:.1f}小時）")
                    continue
                
                time_diffs = [(t - time_values[0]).total_seconds() / 3600 - target_hour for t in time_values]
                time_idx = min(range(len(time_diffs)), key=lambda i: abs(time_diffs[i]))
                
                actual_hours = (time_values[time_idx] - time_values[0]).total_seconds() / 3600
                
                map_file = os.path.splitext(map_output)[0] + f"_{actual_hours:.1f}h.png"
                self.visualize_particles_at_time(ds, time_idx, stats, trajectories, map_file, actual_hours)
                map_files.append((actual_hours, map_file))
            
            report = self._generate_report_content(stats, total_hours, active_points, 
                                                stranded_points, locations, map_files)
            
            return report, map_files
            
        except Exception as e:
            error_msg = f"錯誤：無法生成報告 - {str(e)}"
            print(error_msg)
            return error_msg, []
    
    def _calculate_risk(self, points: np.ndarray, location: Tuple[float, float]) -> str:
        """計算特定位置的風險等級"""
        if not isinstance(points, np.ndarray) or len(points) == 0:
            return "無風險"
        
        try:
            dx = (points[:, 0] - location[0]) * 111 * np.cos(np.radians(location[1]))
            dy = (points[:, 1] - location[1]) * 111
            distances = np.sqrt(dx**2 + dy**2)
            min_distance = np.min(distances)
            
            if min_distance < 5:
                return "高度風險"
            elif min_distance < 10:
                return "中度風險"
            elif min_distance < 20:
                return "輕度風險"
            else:
                return "暫無直接影響"
        except Exception as e:
            print(f"風險計算警告: {e}")
            return "無法評估"
    
    def _get_highest_risk(self, risks: List[str]) -> str:
        """獲取最高風險等級"""
        if not risks:
            return "無風險"
            
        risk_levels = {
            "高度風險": 4,
            "中度風險": 3,
            "輕度風險": 2,
            "暫無直接影響": 1,
            "無風險": 0,
            "無法評估": 0
        }
        
        return max(risks, key=lambda x: risk_levels[x])
    
    def _get_direction(self, start_point: Tuple[float, float], 
                      end_point: Tuple[float, float]) -> str:
        """計算擴散方向"""
        try:
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            
            angle = np.arctan2(dy, dx) * 180 / np.pi
            
            direction_ranges = [
                (-22.5, 22.5, "東向"),
                (22.5, 67.5, "東北向"),
                (67.5, 112.5, "北向"),
                (112.5, 157.5, "西北向"),
                (157.5, 180, "西向"),
                (-180, -157.5, "西向"),
                (-157.5, -112.5, "西南向"),
                (-112.5, -67.5, "南向"),
                (-67.5, -22.5, "東南向")
            ]
            
            for min_angle, max_angle, direction in direction_ranges:
                if min_angle <= angle <= max_angle:
                    return direction
                    
            return "無法判定"
        except Exception as e:
            print(f"方向計算警告: {e}")
            return "無法判定"
        
        
    def _generate_report_content(self, stats: Dict, total_hours: float,
                               active_points: np.ndarray, stranded_points: np.ndarray,
                               locations: List[str], map_files: List[Tuple[float, str]]) -> str:
        """生成報告內容"""
        if len(active_points) > 0 or len(stranded_points) > 0:
            all_points = []
            if len(active_points) > 0:
                all_points.extend(active_points)
            if len(stranded_points) > 0:
                all_points.extend(stranded_points)
            mean_end_point = (np.mean([p[0] for p in all_points]), 
                            np.mean([p[1] for p in all_points]))
            main_direction = self._get_direction(stats['start_point'], mean_end_point)
        else:
            main_direction = "無法判定"
        
        total_particles = stats['active_particles'] + stats['stranded_particles']
        if total_particles == 0:
            print("警告：沒有有效的粒子數據！")
            total_particles = 1
    
        report = self._format_report(
            stats=stats,
            total_hours=total_hours,
            total_particles=total_particles,
            main_direction=main_direction,
            locations=locations,
            map_files=map_files,
            active_points=active_points,
            stranded_points=stranded_points
        )
        
        return report
    
    def _format_report(self, stats: Dict, total_hours: float,
                      total_particles: int, main_direction: str,
                      locations: List[str], map_files: List[Tuple[float, str]],
                      active_points: np.ndarray, stranded_points: np.ndarray) -> str:
        """格式化報告內容"""
        # Convert timestamps to UTC+8
        start_time_utc8 = stats['start_time'].tz_localize('UTC').tz_convert('Asia/Taipei')
        end_time_utc8 = stats['end_time'].tz_localize('UTC').tz_convert('Asia/Taipei')
    
        report = f"""
    油污染擴散影響評估報告
    ====================
    
    一、事件概況
    ----------------
    事故發生時間：{start_time_utc8.strftime('%Y/%m/%d %H:%M')}
    模擬結束時間：{end_time_utc8.strftime('%Y/%m/%d %H:%M')}
    模擬時長：{total_hours:.1f}小時
    油品類型：{stats['oil_type']}
    
    溢油位置：東經{stats['start_point'][0]:.3f}度，北緯{stats['start_point'][1]:.3f}度
    影響範圍：
    - 經度：{stats['lon_range'][0]:.3f}°E - {stats['lon_range'][1]:.3f}°E
    - 緯度：{stats['lat_range'][0]:.3f}°N - {stats['lat_range'][1]:.3f}°N
    
    二、擴散特徵
    ----------------
    主要擴散方向：{main_direction}
    追蹤粒子總數：{total_particles}
    - 海面漂移：{stats['active_particles']}個 ({stats['active_particles']/total_particles*100:.1f}%)
    - 岸際擱淺：{stats['stranded_particles']}個 ({stats['stranded_particles']/total_particles*100:.1f}%)
    
    環境條件：
    - 風速範圍：{stats['wind_speed_range'][0]:.1f} - {stats['wind_speed_range'][1]:.1f} m/s
    - 波高範圍：{stats['wave_height_range'][0]:.1f} - {stats['wave_height_range'][1]:.1f} m
    
    三、各區域影響評估
    ----------------"""
    
        # 評估各地區風險
        high_risk, medium_risk, low_risk, no_risk = [], [], [], []
        
        for location in locations:
            if location in self.locations:
                risks = []
                if len(active_points) > 0:
                    risks.append(self._calculate_risk(active_points, self.locations[location]))
                if len(stranded_points) > 0:
                    risks.append(self._calculate_risk(stranded_points, self.locations[location]))
                
                risk_level = self._get_highest_risk(risks)
                
                if risk_level == "高度風險":
                    high_risk.append(location)
                elif risk_level == "中度風險":
                    medium_risk.append(location)
                elif risk_level == "輕度風險":
                    low_risk.append(location)
                else:
                    no_risk.append(location)
                report += f"\n{location}：{risk_level}"
    
        # 添加應變建議
        report += self._generate_response_suggestions(high_risk, medium_risk, low_risk, no_risk)
    
        # 添加時序說明
        report += "\n\n五、時序擴散說明\n----------------"
        report += "\n不同時間點的擴散特徵："
        
        for hours, map_file in map_files:
            report += f"\n{hours:.1f}小時後 ({os.path.basename(map_file)})："
            current_time = start_time_utc8 + pd.Timedelta(hours=hours)
            report += f"\n- 時間：{current_time.strftime('%Y/%m/%d %H:%M')}"
    
        # 添加備註
        report += """
    \n六、備註
    ----------------
    - 本報告基於數值模擬結果，實際情況可能因天氣、海況等因素而有所差異
    - 建議持續監測實際狀況，並根據現場情況調整應變策略
    - 定期更新模擬結果，及時掌握油污染擴散趨勢"""
    
        return report

    def _generate_response_suggestions(self, high_risk: List[str], medium_risk: List[str],
                                       low_risk: List[str], no_risk: List[str]) -> str:
          """生成應變建議"""
          suggestions = "\n\n四、應變建議\n----------------"
          
          if high_risk:
              suggestions += f"\n高度風險區域（{', '.join(high_risk)}）：\n"
              suggestions += """  - 立即啟動應變機制
    - 部署攔油索及油污清除設備
    - 加強岸際監測
    - 通知相關單位戒備"""
          
          if medium_risk:
              suggestions += f"\n\n中度風險區域（{', '.join(medium_risk)}）：\n"
              suggestions += """  - 預佈應變設備
    - 定期巡查岸際
    - 準備清污設備"""
          
          if low_risk:
              suggestions += f"\n\n輕度風險區域（{', '.join(low_risk)}）：\n"
              suggestions += """  - 持續監測油污動向
    - 建立通報機制
    - 準備應變器材"""
          
          if no_risk:
              suggestions += f"\n\n暫無直接影響區域（{', '.join(no_risk)}）：\n"
              suggestions += "  - 保持警戒\n  - 定期掌握最新動態"
              
          return suggestions
  
    def generate_pdf(self, report_text: str, map_files: List[Tuple[float, str]], 
                    output_pdf: str) -> None:
        """生成包含報告文字和多個時間點地圖的PDF檔案"""
        try:
            # 優化中文字體搜尋
            font_mapping = {
                'microsoft yahei': 'msyh.ttf',
                'simhei': 'simhei.ttf',
                'simsun': 'simsun.ttc',
                'dengxian': 'Deng.ttf',
                'kaiti': 'simkai.ttf'
            }
            
            font_path = self._find_chinese_font(font_mapping)
            font_name = 'Custom-Font' if font_path else 'Helvetica'
            
            if font_path:
                try:
                    pdfmetrics.registerFont(TTFont(font_name, font_path))
                    print(f"成功註冊中文字體: {font_path}")
                except Exception as e:
                    print(f"字體註冊失敗: {str(e)}")
                    font_name = 'Helvetica'
            
            # 創建PDF
            c = canvas.Canvas(output_pdf, pagesize=A4)
            width, height = A4
            
            self._add_title_page(c, width, height, font_name)
            self._add_report_content(c, width, height, font_name, report_text)
            self._add_map_pages(c, width, height, font_name, map_files)
            
            # 儲存PDF
            c.save()
            
        except Exception as e:
            print(f"生成PDF時發生錯誤: {str(e)}")
            raise
      
    def _find_chinese_font(self, font_mapping: Dict[str, str]) -> Optional[str]:
        """搜尋可用的中文字體"""
        system_font_paths = [
            'C:/Windows/Fonts/',  # Windows
            '/usr/share/fonts/',  # Linux
            '/System/Library/Fonts/',  # macOS
            '/Library/Fonts/'  # macOS user fonts
        ]
        
        for base_path in system_font_paths:
            if os.path.exists(base_path):
                for font_file in font_mapping.values():
                    potential_path = os.path.join(base_path, font_file)
                    if os.path.exists(potential_path):
                        return potential_path
        return None
      
    def _add_title_page(self, c: canvas.Canvas, width: float, height: float, 
                       font_name: str) -> None:
        """添加標題頁"""
        c.setFont(font_name, 24)
        title = "油污染擴散影響評估報告"
        title_width = c.stringWidth(title, font_name, 24)
        c.drawString((width - title_width) / 2, height - 100, title)
        
        c.setFont(font_name, 12)
        current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")
        timestamp = f"產製時間：{current_time}"
        time_width = c.stringWidth(timestamp, font_name, 12)
        c.drawString((width - time_width) / 2, height - 150, timestamp)
        
        c.line(50, height - 180, width - 50, height - 180)
        c.showPage()
      
    def _add_report_content(self, c: canvas.Canvas, width: float, height: float,
                          font_name: str, report_text: str) -> None:
        """添加報告內容"""
        y = height - 50
        lines = report_text.split('\n')
        
        for line in lines:
            if y < 50:
                c.showPage()
                y = height - 50
            
            if not line.strip():
                y -= 20
                continue
            
            if '=' in line:  # 主標題
                c.setFont(font_name, 16)
                y -= 30
                line_width = c.stringWidth(line.strip(), font_name, 16)
                c.drawString((width - line_width) / 2, y, line.strip())
            elif line.startswith('----'):  # 分隔線
                c.line(50, y, width - 50, y)
                y -= 20
            else:  # 一般文字
                c.setFont(font_name, 12)
                indent = 50
                if line.startswith('  '):
                    indent += 20
                    line = line.lstrip()
                
                if font_name != 'Helvetica':
                    self._wrap_chinese_text(c, line.strip(), font_name, indent, y, width)
                    y -= 20
                else:
                    wrapped_lines = textwrap.wrap(line.strip(), width=80)
                    for wrapped_line in wrapped_lines:
                        c.drawString(indent, y, wrapped_line)
                        y -= 20
      
    def _wrap_chinese_text(self, c: canvas.Canvas, text: str, font_name: str,
                          x: float, y: float, page_width: float) -> None:
        """處理中文文字自動換行"""
        max_width = page_width - 100
        current_text = ""
        
        for char in text:
            test_text = current_text + char
            if c.stringWidth(test_text, font_name, 12) <= max_width:
                current_text = test_text
            else:
                c.drawString(x, y, current_text)
                y -= 20
                current_text = char
        
        if current_text:
            c.drawString(x, y, current_text)
      
    def _add_map_pages(self, c: canvas.Canvas, width: float, height: float,
                      font_name: str, map_files: List[Tuple[float, str]]) -> None:
        """添加地圖頁面"""
        for hours, map_image in map_files:
            c.showPage()
            try:
                img = ImageReader(map_image)
                img_width, img_height = img.getSize()
                
                scale = min((width - 100) / img_width, (height - 100) / img_height)
                new_width = img_width * scale
                new_height = img_height * scale
                
                x = (width - new_width) / 2
                y = (height - new_height) / 2
                
                c.setFont(font_name, 16)
                map_title = f"油污染擴散分布圖 ({hours:.1f}小時後)"
                title_width = c.stringWidth(map_title, font_name, 16)
                c.drawString((width - title_width) / 2, height - 30, map_title)
                
                c.drawImage(map_image, x, y, width=new_width, height=new_height)
                
            except Exception as e:
                print(f"警告：載入地圖圖片時出錯 ({str(e)})")
                c.drawString(100, height/2, f"無法載入地圖圖片: {str(e)}")





def create_mp4_from_images(image_files: List[str], output_mp4: str, frame_rate: int = 10) -> None:
    """
    將 PNG 圖片序列轉換為 MP4 動畫。
    
    Parameters:
        image_files (List[str]): PNG 文件列表，按照時間順序排序。
        output_mp4 (str): 輸出的 MP4 文件路徑。
        frame_rate (int): 動畫幀率，默認為 10 fps。
    """
    try:
        print(f"開始生成 MP4 動畫: {output_mp4}")
        
        if not image_files:
            raise ValueError("圖片文件列表為空。請提供有效的圖片文件。")
        
        # 確保圖片文件列表是排序的
        image_files_sorted = sorted(image_files)
        
        # 獲取圖片目錄的公共路徑
        common_path = os.path.commonpath(image_files_sorted)
        
        # 構建輸入模式，使用 os.path.join 確保路徑分隔符正確
        input_pattern = os.path.join(common_path, "frame_h%03d.png")
        
        # 確認圖片命名從 000 開始
        first_image = os.path.basename(image_files_sorted[0])
        if not first_image.startswith("frame_h") or not first_image.endswith(".png"):
            raise ValueError("圖片文件名應該以 'frame_h' 開頭並以 '.png' 結尾。")
        
        # 構建 ffmpeg 命令，添加 -start_number 0 和 scale 過濾器
        command = [
            "ffmpeg", "-y",
            "-framerate", str(frame_rate),       # 設定幀率
            "-start_number", "0",                # 指定起始幀編號
            "-i", input_pattern,                 # 輸入圖片序列模式
            "-vf", "scale=ceil(iw/2)*2:ceil(ih/2)*2",  # 調整尺寸為偶數
            "-c:v", "libx264",                   # 使用 H.264 編碼
            "-pix_fmt", "yuv420p",               # 確保兼容性
            output_mp4                           # 輸出 MP4 文件
        ]
        
        # 打印命令以便調試
        print("執行命令:", ' '.join(command))
        
        # 調用 ffmpeg 命令，捕捉標準輸出和錯誤輸出
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 打印 ffmpeg 的標準輸出
        print(result.stdout.decode('utf-8'))
        
        print(f"MP4 動畫已成功生成: {output_mp4}")
    except subprocess.CalledProcessError as e:
        # 打印 ffmpeg 的錯誤輸出
        print(f"生成 MP4 動畫時發生錯誤: {e.stderr.decode('utf-8')}")
    except Exception as e:
        print(f"生成 MP4 動畫時發生錯誤: {str(e)}")


'''

def create_mp4_from_images(image_files: List[str], output_mp4: str, frame_rate: int = 10) -> None:
    """
    將 PNG 圖片序列轉換為 MP4 動畫。
    
    Parameters:
        image_files (List[str]): PNG 文件列表，按照時間順序排序。
        output_mp4 (str): 輸出的 MP4 文件路徑。
        frame_rate (int): 動畫幀率，默認為 10 fps。
    """
    try:
        print(f"開始生成 MP4 動畫: {output_mp4}")
        
        # 使用 ffmpeg 將圖片序列轉換為 MP4
        # 確保 image_files 的命名遵循某種順序（如 frame_h000.png, frame_h001.png...）
        # 如果文件命名不符合，請提前處理成按順序命名的格式。
        
        input_pattern = os.path.commonpath(image_files) + "\frame_h%03d.png"  # 假設命名為 frame_000.png
        command = [
            "ffmpeg", "-y",
            "-framerate", str(frame_rate),       # 設定幀率
            "-start_number", "0",                # 指定起始幀編號
            "-i", input_pattern,                # 輸入圖片序列模式
            "-c:v", "libx264",                  # 使用 H.264 編碼
            "-pix_fmt", "yuv420p",              # 確保兼容性
            output_mp4                          # 輸出 MP4 文件
        ]
        
        # 調用 ffmpeg 命令
        subprocess.run(command, check=True)
        
        print(f"MP4 動畫已成功生成: {output_mp4}")
    except Exception as e:
        print(f"生成 MP4 動畫時發生錯誤: {str(e)}")

def main():
    try:
        # 設置檔案路徑
        base_dir = "D:/opendrift/20241114_oil_spill_test_T_ocm_1114"
        nc_file = os.path.join(base_dir, "combined_output.nc")
        map_output = os.path.join(base_dir, "oil_spill_map.png")
        report_output = os.path.join(base_dir, "oil_impact_report.txt")
        pdf_output = os.path.join(base_dir, "oil_impact_report.pdf")
        animation_dir = os.path.join(base_dir, "animation")
        gif_output = os.path.join(animation_dir, "oil_spill_animation.gif")
        
        # 確保輸出目錄存在
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(animation_dir, exist_ok=True)
        
        # 設置地圖樣式
        map_style = MapStyle(
            use_high_res=True, 
            map_style='osm_street',
            tile_zoom=9
        )
        
        # 生成報告
        print("開始生成油污染擴散影響評估報告...")
        report_generator = Report(map_style=map_style)
        
        # 首先分析NC檔案以取得必要資訊
        stats, active_points, stranded_points, trajectories, ds = report_generator.analyze_nc_file(nc_file)
        
        # 生成每小時的模擬圖
        hourly_map_base = os.path.join(animation_dir, "frame")
        hourly_files = report_generator.generate_hourly_maps(
            ds, stats, trajectories, hourly_map_base
        )
        
        # 生成GIF動畫
        report_generator.create_gif(hourly_files, gif_output)
        
        # 生成報告和固定時間點的地圖
        report, map_files = report_generator.generate(
            nc_file,
            [
                '台北港、淡水河口',
                '林口發電廠',
                '大潭三接港',
                '新北市三芝海域'
            ],
            map_output
        )
        
        # 儲存文字報告
        print(f"\n儲存報告至：{report_output}")
        with open(report_output, "w", encoding="utf-8") as f:
            f.write(report)
            
        # 生成PDF報告
        print(f"生成PDF報告：{pdf_output}")
        report_generator.generate_pdf(report, map_files, pdf_output)
        
        print("\n=== 處理完成 ===")
        print(f"- 報告檔案：{report_output}")
        print(f"- PDF報告：{pdf_output}")
        print(f"- 動畫檔案：{gif_output}")
        print("- 固定時間點地圖檔案：")
        for hours, map_file in map_files:
            print(f"  - {hours:.1f}小時: {map_file}")
        
    except Exception as e:
        print(f"程式執行錯誤：{str(e)}")

if __name__ == "__main__":
    main()
    '''
    
def main():
    try:
        # 設置檔案路徑
        base_dir = "D:/opendrift/20241114_oil_spill_test_T_ocm_1114"
        nc_file = os.path.join(base_dir, "combined_output.nc")
        map_output = os.path.join(base_dir, "oil_spill_map.png")
        report_output = os.path.join(base_dir, "oil_impact_report.txt")
        pdf_output = os.path.join(base_dir, "oil_impact_report.pdf")
        animation_dir = os.path.join(base_dir, "animation")
        gif_output = os.path.join(animation_dir, "oil_spill_animation.gif")
        mp4_output = os.path.join(animation_dir, "oil_spill_animation.mp4")  
        
        # 確保輸出目錄存在
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(animation_dir, exist_ok=True)
        
        # 設置地圖樣式
        map_style = MapStyle(
            use_high_res=True, 
            map_style='osm_street',
            tile_zoom=9
        )
        
        # 生成報告
        print("開始生成油污染擴散影響評估報告...")
        report_generator = Report(map_style=map_style)
        
        # 首先分析NC檔案以取得必要資訊
        stats, active_points, stranded_points, trajectories, ds = report_generator.analyze_nc_file(nc_file)
        
        # 生成每小時的模擬圖
        hourly_map_base = os.path.join(animation_dir, "frame")
        hourly_files = report_generator.generate_hourly_maps(
            ds, stats, trajectories, hourly_map_base
        )
        
        # 生成GIF動畫
        report_generator.create_gif(hourly_files, gif_output)
        
        # 生成MP4動畫
        create_mp4_from_images(hourly_files, mp4_output)
        
        # 生成報告和固定時間點的地圖
        report, map_files = report_generator.generate(
            nc_file,
            [
                '台北港、淡水河口',
                '林口發電廠',
                '大潭三接港',
                '新北市三芝海域'
            ],
            map_output
        )
        
        # 儲存文字報告
        print(f"\n儲存報告至：{report_output}")
        with open(report_output, "w", encoding="utf-8") as f:
            f.write(report)
            
        # 生成PDF報告
        print(f"生成PDF報告：{pdf_output}")
        report_generator.generate_pdf(report, map_files, pdf_output)
        
        print("\n=== 處理完成 ===")
        print(f"- 報告檔案：{report_output}")
        print(f"- PDF報告：{pdf_output}")
        print(f"- GIF動畫檔案：{gif_output}")
        print(f"- MP4動畫檔案：{mp4_output}")
        print("- 固定時間點地圖檔案：")
        for hours, map_file in map_files:
            print(f"  - {hours:.1f}小時: {map_file}")
        
    except Exception as e:
        print(f"程式執行錯誤：{str(e)}")

if __name__ == "__main__":
    main()
