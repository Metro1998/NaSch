# NaSch
A traffic flow simulation model 

# Criterion
criterion1_acceleration: 
vn ——> min(vn + 1, vmax), the driver is intended to drive as fast as vmax.

criterion2_deceleration: 
vn ——> min(vn, dn), the driver decelerates to avoid crash.
vn = 

criterion3_randomization_probability: 
p, vn ——> max(vn - 1, 0), cars decelerates with possibility p due certain reasons.

criterion4_movement: 
xn ——> xn + vn, cars move ahead according to the velocity.




