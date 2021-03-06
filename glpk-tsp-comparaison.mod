/**
 * filename : glpk-tsp-comparaison.mod 
 * command  : glpsol --math glpk-tsp-comparaison.mod 
 *
 * Solution optimale : 1-5-6-4-10-9-8-3-7-2 (dist=2826.498405)
 * 
 */
/* parameters */
param n ;

set I := (1..n) ;

param d {i in I, j in I};

/* Decision variables */
var x {i in I, j in I} binary ;
var u {i in I}, >=1, <=n, integer ;

/* Objective function */
minimize z: sum{i in I, j in I : i!=j} d[i,j]*x[i,j] ;

/* Constraints */
s.t. totown{j in I}   : sum{i in I : i!=j} x[i,j] = 1 ;
s.t. fromtown{i in I} : sum{j in I : i!=j} x[i,j] = 1 ;
s.t. const_u1          : u[1] = 1 ;		
s.t. tour{i in I, j in I : j>=2 and i!=j} : u[i]-u[j]+n*x[i,j] <= n-1 ; 

solve ; 

display u ;
printf {i in I, j in I : x[i,j]>0} : "de ville %d a ville %d\n", i, j ;
printf "dist.tot=%f\n",z;

/** Data section */
data;

param n := 16;

/* La matrice doit ?tre ins?r?e/modifi?e ? la main*/
param d : 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16:=
1  -1.0  666.1  281.1  395.6  291.2  326.3  640.8  426.9  600.2  561.5 1041.0  655.0  975.0 1120.8  299.0  260.0  
2 666.1   -1.0  649.3 1047.1  945.1  978.1   45.0  956.2 1135.0 1133.0 1638.8 1258.6 1440.1 1515.7  957.8  724.0  
3 281.1  649.3   -1.0  603.5  508.9  542.5  610.6  308.1  485.6  487.3 1266.7  891.4 1247.8 1399.7  504.9  537.4  
4 395.6 1047.1  603.5   -1.0  104.4   69.6 1026.4  525.0  611.0  533.9  663.2  294.4  711.1  897.0  100.1  384.2   
5 291.2  945.1  508.9  104.4   -1.0   35.4  923.6  470.6  583.6  513.5  760.8  382.4  769.0  944.3   25.0  309.2  
6 326.3  978.1  542.5   69.6   35.4   -1.0  957.0  491.6  596.0  523.3  726.1  349.3  744.2  922.8   40.3  328.8  
7 640.8   45.0  610.6 1026.4  923.6  957.0   -1.0  918.1 1095.9 1095.7 1627.4 1245.2 1440.3 1521.7  935.4  713.9  
8 426.9  956.2  308.1  525.0  470.6  491.6  918.1   -1.0  183.4  180.3 1144.9  812.0 1234.3 1414.2  452.5  661.0  
9 600.2 1135.0  485.6  611.0  583.6  596.0 1095.9  183.4   -1.0   83.2 1165.6  873.9 1316.8 1507.1  561.5  818.0  
10 561.5 1133.0  487.3  533.9  513.5  523.3 1095.7  180.3   83.2   -1.0 1082.6  792.1 1236.6 1428.3  490.4  763.7 
11 1041.0 1638.8 1266.7  663.2  760.8  726.1 1627.4 1144.9 1165.6 1082.6   -1.0  387.1  442.7  619.6  762.4  914.8
12 655.0 1258.6  891.4  294.4  382.4  349.3 1245.2  812.0  873.9  792.1  387.1   -1.0  452.1  653.2  388.1  537.7 
13 975.0 1440.1 1247.8  711.1  769.0  744.2 1440.3 1234.3 1316.8 1236.6  442.7  452.1   -1.0  205.5  784.1  759.3 
14 1120.8 1515.7 1399.7  897.0  944.3  922.8 1521.7 1414.2 1507.1 1428.3  619.6  653.2  205.5   -1.0  961.7  883.9
15 299.0  957.8  504.9  100.1   25.0   40.3  935.4  452.5  561.5  490.4  762.4  388.1  784.1  961.7   -1.0  332.4
16 260.0  724.0  537.4  384.2  309.2  328.8  713.9  661.0  818.0  763.7  914.8  537.7  759.3  883.9  332.4   -1.0;

end ;

