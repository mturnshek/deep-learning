function brain(num_inputs, num_actions) {
	let temporal_window = 0; // amount of temporal memory. 0 = agent lives in-the-moment :)
	let network_size = num_inputs*temporal_window + num_actions*temporal_window + num_inputs;

	// the value function network computes a value of taking any of the possible actions
	// given an input state. Here we specify one explicitly the hard way
	// but user could also equivalently instead use opt.hidden_layer_sizes = [20,20]
	// to just insert simple relu hidden layers.
	let layer_defs = [];
	layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:network_size});
	layer_defs.push({type:'fc', num_neurons: 100, activation:'relu'});
	//layer_defs.push({type:'fc', num_neurons: 100, activation:'relu'});
	layer_defs.push({type:'regression', num_neurons:num_actions});

	// options for the Temporal Difference learner that trains the above net
	// by backpropping the temporal difference learning rule.
	let tdtrainer_options = {learning_rate:0.001, momentum:0.0, batch_size:16, l2_decay:0.01};

	let opt = {};
	opt.temporal_window = temporal_window;
	opt.experience_size = 100000;
	opt.start_learn_threshold = 1000;
	opt.gamma = 0.8;
	opt.learning_steps_total = 8000000;
	opt.learning_steps_burnin = 50000;
	opt.epsilon_min = 0.05;
	opt.epsilon_test_time = 0.05;
	opt.layer_defs = layer_defs;
	opt.tdtrainer_options = tdtrainer_options;

	return new deepqlearn.Brain(num_inputs, num_actions, opt); // woohoo
}

class Environment {
	constructor() {
		// Make an instance of two and place it on the page.
		let elem = document.getElementById('canvas');
		this.two = new Two({fullscreen: true});
		(this.two).appendTo(elem);

		this.width = this.two.width;
		this.height = this.two.height;

		let num_red = 1;
		let num_green = 3;

		this.balls = []
		this.green_balls = []
		this.red_balls = []

		let id = 0;
		let ball = null;
		let x = 0;
		let y = 0;

		this.state_dim = (num_red + num_green) * 2; // x, y for red, green

		// Create the predators
		let i = 0;
		for (i = 0; i < num_red; i++) {
			x = 50; // make random
			y = 50; // make random
			ball = new Ball(x, y, 'red', id, this);
			this.balls.push(ball);
			this.red_balls.push(ball);
			id++;
		}

		// Create the prey
		for (i = 0; i < num_green; i++) {
			x = 500; // make random
			y = 50; // make random
			ball = new Ball(x, y, 'green', id, this);
			this.balls.push(ball);
			this.green_balls.push(ball);
			id++;
		}

		let water_x = 1100;
		let water_y = 0;
		this.water = new Ball(water_x, water_y, 'blue', id, this);
	}


	// Given the id (index) of the ball, return the vector representing its state
	perception(ball_id) {
		let self_perception = [];
		let others_perception = [];

		let i = 0;
		for (i = 0; i < this.balls.length; i++) {
			let ball = this.balls[i];

			// If the ball is the perceiving agent, (ball.id == ball_id)
			if (ball.id == ball_id) {
				self_perception.push(ball.x);
				self_perception.push(ball.y);
			} else {
				others_perception.push(ball.x);
				others_perception.push(ball.y);
			}
		}
		
		return self_perception.concat(others_perception);
	}


	get_reward(ball_id) {
		let reward = 0.0;
		let ball = this.balls[ball_id];

		let red_predator_reward = 2.0;
		let green_prey_reward = -10.0;
		let water_reward = 1.0;

		let red_balls = this.red_balls;
		let green_balls = this.green_balls;

		if (ball.color == 'red') {
			let i = 0;
			for (i = 0; i < green_balls.length; i++) {
				if (ball.touching(green_balls[i])) {
					reward += red_predator_reward;
				}
			}
		} else {
			let i = 0;
			for (i = 0; i < red_balls.length; i++) {
				if (ball.touching(red_balls[i])) {
					reward += green_prey_reward;
				}
			}
			if (ball.touching(this.water)) {
				reward += water_reward;
			}
		}

		return reward;
	}


	update() {
		let ball = null;
		let i = 0;
		for (i = 0; i < this.balls.length; i++) {
			ball = this.balls[i];
			ball.update();
		}
		this.two.update();
	}


	start() {
		let tick_time = 0.5;
		setInterval(this.update.bind(this), tick_time);
	}
}


///////////////////////

///////////////////////

///////////////////////

///////////////////////

///////////////////////

///////////////////////

///////////////////////

///////////////////////

///////////////////////

class Ball {
	constructor(x, y, color, id, environment) {
		this.environment = environment;
		this.id = id;
		this.color = color;

		if (color == 'red') {
			this.radius = 40.0;
			this.sentient = true;
			this.brain = brain(environment.state_dim, 4);
		} else if (color == 'green') {
			this.radius = 10.0;
			this.sentient = true;
			this.brain = brain(environment.state_dim, 4);
		} else {
			this.radius = 300.0;
			this.sentient = false;
			this.brain = null;
		}

		// Position and movement
		this.x = x;
		this.y = y;
		this.dx = 0.0;
		this.dy = 0.0;
		this.acceleration = 0.25;
		this.top_speed = 5.0;

		// Drawing
		let red = "#922";
		let red_line = "000";
		let green = "#292";
		let green_line = "000";
		let blue = "#04C";

		this.circle = environment.two.makeCircle(x, y, this.radius);
		this.circle.opacity = 0.75;
		if (this.color == 'red') {
			this.circle.fill = red;
			this.circle.stroke = red_line;
			this.circle.linewidth = 20;
		} else if (this.color == 'green') {
			this.circle.fill = green;
			this.circle.stroke = green_line;
			this.circle.linewidth = 20;
		} else {
			this.circle.fill = blue;
			this.circle.stroke = blue;
			this.circle.linewidth = 0;
		}

	}


	// Movement
	go_left() {
		if (this.dx > -this.top_speed) {
			this.dx -= this.acceleration;
		}
	}


	go_right() {
		if (this.dx < this.top_speed) {
			this.dx += this.acceleration;
		}
	}


	go_up() {
		if (this.dy > -this.top_speed) {
			this.dy -= this.acceleration;
		}
	}


	go_down() {
		if (this.dy < this.top_speed) {
			this.dy += this.acceleration;
		}
	}


	move() {
		if (this.x > this.environment.width) {
			this.x = this.environment.width;
			this.dx = 0;
		} else if (this.x < 0) {
			this.x = 0;
			this.dx = 0;
		}
		if (this.y > this.environment.height) {
			this.y = this.environment.height;
			this.dy = 0;
		} else if (this.y < 0) {
			this.y = 0;
			this.dy = 0;
		}

		this.x += this.dx;
		this.y += this.dy;
	}


	decide(action) {
		if (action == 0) {
			this.go_left();
		} else if (action == 1) {
			this.go_right();
		} else if (action == 2) {
			this.go_up();
		} else {
			this.go_down();
		}
	}


	touching(ball) {
		let distance = Math.sqrt(Math.pow(this.x - ball.x, 2) + Math.pow(this.y - ball.y, 2))
		let radii = this.radius + ball.radius;
		if (distance < radii) {
			return true;
		} else {
			return false;   
		}
	}


	draw() {
		this.circle.translation.set(this.x, this.y);
	}


	update() {
		if (this.sentient) {
			// Choose an action
			let state = [];
			let action = this.brain.forward(state);

			// Update the environment
			this.decide(action);
			this.move();
			this.draw();

			// Reward the action
			let reward = this.environment.get_reward(this.id);

			this.brain.backward(reward);
		}
	}
}

///////////////////////

///////////////////////

///////////////////////

///////////////////////

///////////////////////

///////////////////////

///////////////////////

///////////////////////

///////////////////////


document.addEventListener("DOMContentLoaded", function(event) { 
	let env = new Environment();
	env.start();
});