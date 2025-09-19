# Simplex and branch & bound algorithms

## Example of run :
cargo run --release mcdo.lp

This command above start the program in command mode and solve the linear program written in mcdo.lp
Branch and bound algorithm is used only if there are some integer variables.

## Start in server mode
cargo run --release server

This command above start the program in server mode (0.0.0.0:8888). There are two routes : /simplex in post method and branch_and_bound in post method. This two routes takes a .lp file and solves it.

## Executing time of branch and bound algorithm on n-queens problem (8 <= n <= 40) with 70.0 seconds max timeout.
<img width="1000" height="600" alt="nqueens" src="https://github.com/Wronskode/Simplex/blob/main/nqueens.png" />
