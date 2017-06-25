function brain(state_dim, num_actions) {
	let env = {};
	env.getNumStates = function() { return state_dim; }
	env.getMaxNumActions = function() { return num_actions; }

	let spec = {};
	spec.update = 'qlearn'; // qlearn | sarsa
	spec.gamma = 0.99; // discount factor, [0, 1)
	spec.epsilon = 0.99; // initial epsilon for epsilon-greedy policy, [0, 1)
	spec.experience_add_every = 1; // number of time steps before we add another experience to replay memory
	spec.experience_size = 10000; // size of experience
	spec.learning_steps_per_iteration = 20;
	spec.tderror_clamp = 1.0; // for robustness
	spec.num_hidden_units = 256 // number of neurons in hidden layer

	return new RL.DQNAgent(env, spec);
}


function epsilon_decay(x) {
	return x * .9999;
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

		this.state_dim = (num_red + num_green) * 4; // x, y, dx, dy for red, green

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
				self_perception.push(ball.dx);
				self_perception.push(ball.dy);
			} else {
				others_perception.push(ball.x);
				others_perception.push(ball.y);
				others_perception.push(ball.dx);
				others_perception.push(ball.dy);
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
			let action = this.brain.act(state);
			
			// Update the environment
			this.decide(action);
			this.move();
			this.draw();

			// Reward the action
			let reward = this.environment.get_reward(this.id);

			this.brain.learn(reward);

			// Reduce epsilon
			this.brain.epsilon = epsilon_decay(this.brain.epsilon);
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