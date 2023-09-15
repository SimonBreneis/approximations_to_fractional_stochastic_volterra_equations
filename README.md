For obtaining good Markovian approximations (good quadrature rules), call 
nodes, weights = RoughKernel.quadrature_rule(H, T, N)
where H is the Hurst parameter, T the maturity and N the approximating dimension (N=2 or N=3 is usually enough).

For efficiently simulating the Markovian approximation of the rough Heston model, call
rHestonMarkovSimulation.samples()
where one uses the nodes and weights generated before.
