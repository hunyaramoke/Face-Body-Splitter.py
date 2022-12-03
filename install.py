import launch

if not launch.is_installed("glob"):
    launch.run_pip("install glob")
    
if not launch.is_installed("openmim"):
    launch.run_pip("install openmim")
if not launch.is_installed("mmcv-full"):
    launch.run_pip("install mmcv-full")
if not launch.is_installed("mmdet"):
    launch.run_pip("install mmdet")
if not launch.is_installed("mmpose"):
    launch.run_pip("install mmpose")

if not launch.is_installed("anime-face-detector"):
    launch.run_pip("install anime-face-detector")