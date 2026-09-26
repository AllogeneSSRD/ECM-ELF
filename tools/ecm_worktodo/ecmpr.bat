C:/anaconda3/envs/web/python.exe "D:\code\GIMPS\GIMPS_同步\ECM\pipeline\ECM.py" ^
--append-prmers ^
--set-b1 100000 --set-b2 0 --skip-curves 0 --gpu-curves 50 --sort-by n ^
--input        "D:\code\GIMPS\GIMPS_同步\ECM\pipeline\sorted.csv" ^
--out-prmers   "D:\code\GIMPS\prmers\prmers-windows_v4.18.2\worktodo.txt" ^
--out-windows  "D:\code\GIMPS\p95v3104\worktodo_add.csv" ^
--save-pattern "resume_p{n}_ECM_TE_B1_{b1}.p95"

C:/anaconda3/envs/web/python.exe "D:\code\GIMPS\GIMPS_同步\ECM\pipeline\ECM.py" ^
--append-windows --append-prmers ^
--set-b1 100000 --set-b2 0 --skip-curves 0 --gpu-curves 50 --sort-by n ^
--input        "D:\code\GIMPS\GIMPS_同步\ECM\pipeline\sorted.csv" ^
--out-windows  "D:\code\GIMPS\p95v3104\worktodo_save.csv" ^
--save-pattern "resume_p{n}_ECM_TE_B1_{b1}.p95"
@REM --out-windows  "D:\code\GIMPS\GIMPS_同步\ECM\_pr-p95-s2.csv" ^
@REM --out-prmers      "D:\code\GIMPS\GIMPS_同步\ECM\_pr-s1.csv" ^
