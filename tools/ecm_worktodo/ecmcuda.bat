C:/anaconda3/envs/web/python.exe "d:\code\GIMPS\GIMPS_同步\ECM\pipeline\ECM.py" ^
--append-linux --append-windows ^
--set-b1 110e6 --gmpecm-b2 0 --p95-b2 0 --skip-curves 0 --gpu-curves 960 --sort-by n ^
--input "d:\code\GIMPS\GIMPS_同步\ECM\pipeline\sorted.csv" ^
--out-windows  "D:\code\GIMPS\GIMPS_同步\worktodo_add.csv"

C:/anaconda3/envs/web/python.exe "d:\code\GIMPS\GIMPS_同步\ECM\pipeline\ECM.py" ^
--append-linux --append-windows ^
--set-b1 110e6 --gmpecm-b2 0 --p95-b2 0 --skip-curves 0 --gpu-curves 960 --sort-by n ^
--input "d:\code\GIMPS\GIMPS_同步\ECM\pipeline\sorted.csv" ^
--out-windows  "D:\code\GIMPS\GIMPS_同步\worktodo_save.csv"

@REM --out-linux "D:\code\GIMPS\gmp-ecm\gpu\worktodo.save.sh" ^
