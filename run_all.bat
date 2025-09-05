@echo off
setlocal

rem ==== 方法与轨迹组合 ====
for %%M in (adaptive pid nnadaptive) do (
  for %%T in (fig8 circle ellipse) do (
    echo Running method=%%M traj=%%T ...
    python data_collection_all.py --mode test --method %%M --traj %%T
  )
)

rem ==== 运行 pinn_online_adaptive_0903.py 三次，每个 traj 后删除 warm_start ====
for %%T in (fig8 circle ellipse) do (
  echo Running pinn_online_adaptive_0903.py with traj=%%T ...
  python pinn_online_adaptive_0903.py --mode test --traj %%T
  python pinn_online_adaptive_0903.py --mode test --traj %%T
  python pinn_online_adaptive_0903.py --mode test --traj %%T

  if exist warm_start (
    echo Deleting warm_start ...
    rmdir /s /q warm_start
  )
)

endlocal
