x = linspace(1,200,200);
time = 4.0 .* x / 200 ;
signal = 0.2 + 0.5 * sin(time) + 0.2 * cos(10*time) + 0.2 * sin(4*time);

rf = rainflow(signal)