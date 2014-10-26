/**
 * @file mass_spring.cpp
 * Implementation of mass-spring system using Graph
 *
 * @brief Reads in two files specified on the command line.
 * First file: 3D Points (one per line) defined by three doubles
 * Second file: Tetrahedra (one per line) defined by 4 indices into the point
 * list
 */

#include <fstream>

#include "CS207/SDLViewer.hpp"
#include "CS207/Util.hpp"
#include "CS207/Color.hpp"

#include "Graph.hpp"
#include "Point.hpp"


// Gravity in meters/sec^2
static constexpr double grav = 9.81;

//To store initial length of Edges for Problem1Force
double Initial_length;

/** Custom structure of data to store with Nodes */
struct NodeData {
	Point velocity;  //< Node velocity
	double mass;     //< Node mass
};

/** Custom structure of data to store with Edges */
struct EdgeData {
	double length;   //< Edge length
};

// Define your Graph type
typedef Graph<NodeData, EdgeData> GraphType;
typedef typename GraphType::node_type Node;
typedef typename GraphType::edge_type Edge;

struct FixedConstraint;

/** Change a graph's nodes according to a step of the symplectic Euler
 *    method with the given node force.
 * @param[in,out] g      Graph
 * @param[in]     t      The current time (useful for time-dependent forces)
 * @param[in]     dt     The time step
 * @param[in]     force  Function object defining the force per node
 * @param[in]	  constrain Constrain object defining the constraints for the simulation
 *                Defaulted to FixedConstraint where the Points (0,0,0) and (1,0,0) are fixed
 * @return the next time step (usually @a t + @a dt)
 *
 * @tparam G::node_value_type supports ???????? YOU CHOOSE
 * @tparam F is a function object called as @a force(n, @a t),
 *           where n is a node of the graph and @a t is the current time.
 *           @a force must return a Point representing the force vector on Node
 *           at time @a t.
 */
template <typename G, typename F, typename C = FixedConstraint>
double symp_euler_step(G& g, double t, double dt, F force, C constrain = C() ) {
	// Compute the {n+1} node positions
	for (auto it = g.node_begin(); it != g.node_end(); ++it) {
		auto n = *it;

		// Update the position of the node according to its velocity
		// x^{n+1} = x^{n} + v^{n} * dt
		n.position() += n.value().velocity * dt;
	}

	// Compute the {n+1} node velocities
	for (auto it = g.node_begin(); it != g.node_end(); ++it) {
		auto n = *it;
			// v^{n+1} = v^{n} + F(x^{n+1},t) * dt / m
				n.value().velocity += force(n, t) * (dt / n.value().mass);
	}

	//Set constraints
	constrain(g, t);

	return t + dt;
}


/** Force function object for HW2 #1. */
struct Problem1Force {
	/** Return the force being applied to @a n at time @a t.
	 *
	 * For HW2 #1, this is a combination of mass-spring force and gravity,
	 * except that points at (0, 0, 0) and (1, 0, 0) never move. We can
	 * model that by returning a zero-valued force. */
	double K = 100;
	Point operator()(Node n, double t) {
		(void) t;     // silence compiler warnings

		//Fixed at (0,0,0) and (1,0,0)
		if (n.position() == Point(0 ,0 ,0) || n.position() == Point(1 ,0 ,0)) {
			return Point(0,0,0);
		}
		else {
			Point springForce = Point(0,0,0);
			for (auto it = n.edge_begin(); it != n.edge_end(); ++it) {
				Point incidentNode = (*it).node2().position();
				Point p_distance = n.position() - incidentNode;
				double distance = norm(p_distance);
				springForce += (-1.0)* K * (p_distance)/distance * (distance - (*it).value().length );
			}
			return springForce + n.value().mass*Point(0,0,-1.0*grav);
		}

	}
};

//Other forces

/** Gravity Force Functor that returns the Gravity Force
 */
struct GravityForce {
	double gravity_;
	/** GravityForce Constructor.
	 * @param[in] g Gravity in m/s^2.
	 */
	GravityForce(double g = grav) : gravity_(g) {}

	/** Calculates Gravity Force
	 * @param[in] n Valid node.
	 * @param[in] t Valid time.
	 * @return Point object that represents the gravity force calculated as m*g.
	 */
	Point operator()(Node n, double t) {
		(void) t;     // silence compiler warnings
		return n.value().mass * Point(0,0,-1.0 * gravity_);
	}
};

/** Spring Force Functor that returns the Spring Force
 */
struct MassSpringForce {
	double K_;
	/** MassSpringForce Constructor.
	 * @param[in] K Spring constant in N/m
	 */
	MassSpringForce(double K = 100) : K_(K) {}

	/** Calculates Mass Spring Force
	 * @param[in] n Valid node.
	 * @param[in] t Valid time.
	 * @return Point object that represents the mass spring force.
	 */
	Point operator()(Node n, double t) {
		(void) t;     // silence compiler warnings
		Point springForce = Point(0,0,0);
		for (auto it = n.edge_begin(); it != n.edge_end(); ++it) {
			Point incidentNode = (*it).node2().position();
			Point p_distance = n.position() - incidentNode;
			double distance = norm(p_distance);
			springForce += (-1.0) * K_ * (p_distance)/distance * (distance - (*it).value().length );
		}
		return springForce;
	}
};

/** Damping Force Functor that returns the Damping Force
 * @param[in] damping_const Damping constant in N*s/m
 */
struct DampingForce {
	double damping_const_;

	/** DampingForce Constructor.
	 * @param[in] damping_const Damping constant in N*s/m
	 */
	DampingForce(double damping_const) : damping_const_(damping_const) {}

	/** Calculates Damping Force
	 * @param[in] n Valid node.
	 * @param[in] t Valid time.
	 * @return Point object that represents the damping force calculated as damping_const*velocity.
	 */
	Point operator()(Node n, double t) {
		(void) t;     // silence compiler warnings
		return -1.0*damping_const_*n.value().velocity;
	}
};

/** Combine Force Functor that returns a combination of forces
 * @param[in] Two valid forces in f1 and f2.
 */
template <typename Force1, typename Force2>
struct CombineForces {
	Force1 f1_;
	Force2 f2_;

	/** CombineForces Constructor.
	 * @param[in] f1 First valid force.
	 * @param[in] f2 Second valid force.
	 */
	CombineForces(Force1 f1, Force2 f2)
			: f1_(f1), f2_(f2) {}

	/** Calculates Combine Forces
	 * @param[in] n Valid node.
	 * @param[in] t Valid time.
	 * @return Point object that represents the combination of forces of @a f1_ and @a f2_.
	 */
	Point operator() (Node n, double t){
		return f1_(n,t) + f2_(n,t);
	}
};

/** Combine Force Function that returns a combination of forces
 * @param[in] Two valid forces in f1, and f2.
 * @pre Valid forces that input a node and time with a function call of f(n,t).
 * @return A CombineForces object that adds up the forces.
 */
template <typename Force1, typename Force2>
CombineForces<Force1, Force2> make_combined_force(Force1 f1, Force2 f2){
	return CombineForces<Force1, Force2> ({f1, f2});
}

/** Combine Force Function that returns a combination of forces
 * @param[in] Three valid forces in f1, f2, and f3.
 * @pre Valid forces that input a node and time with a function call of f(n,t).
 * @return A CombineForces object that adds up the forces.
 */
template <typename Force1, typename Force2, typename Force3>
CombineForces<CombineForces<Force1, Force2>, Force3> make_combined_force(Force1 f1, Force2 f2, Force3 f3){
	return CombineForces<CombineForces<Force1, Force2>, Force3> (CombineForces<Force1, Force2>(f1,f2), f3);
}

/** Null Constraint
 */
struct NullConstraint {

	/** Null Constraint Setter
	 * @param[in] g Valid graph.
	 * @param[in] t Valid time.
	 * @return Point object that represents the combination of forces of @a f1_ and @a f2_.
	 */
	void operator()(GraphType& g, double t) {
		(void) t;     // silence compiler warnings
		(void) g;
	}

};

/** Fixed Constraint where Point(0,0,0) and Point(1,0,0) are static
 */
struct FixedConstraint {

	/** Fixed Constraint Setter
	 * @param[in] g Valid graph.
	 * @param[in] t Valid time.
	 * @post The velocity of Point(0,0,0) and Point(1,0,0) are 0.
	 */
	void operator()(GraphType& g, double t) {
			(void) t;     // silence compiler warnings
			for(auto it = g.node_begin(); it != g.node_end(); ++it) {
				Node n = (*it);
				if(n.position() == Point(0,0,0) || n.position() == Point(1,0,0)) {
					n.value().velocity =  Point(0,0,0);
				}
			}
		}
};

/** Horizontal Plane Constraint that models an impassable plane.
 */
struct HPlaneConstraint {
double z_constraint_; //coordinate for the horizontal plane

	/** HPlaneConstraint Constructor.
	 * @param[in] z_constraint Sets the z-coordinate to define the horizontal plane.
	 */
	HPlaneConstraint(double z_constraint) : z_constraint_(z_constraint) {}

	/** Horizontal Constraint Setter
	 * @param[in] g Valid graph.
	 * @param[in] t Valid time.
	 * @post The velocity of @a z_constraint_ is 0 and is set to the closest point to
	 *       the horizontal plane as defined by @a z_constraint_.
	 */
	void operator()(GraphType& g, double t) {
		(void) t;     // silence compiler warnings
		for(auto it = g.node_begin(); it != g.node_end(); ++it) {
			Node n = (*it);
			if(n.position().z < z_constraint_) {
				n.position().z = z_constraint_;
				n.value().velocity.z = 0;
			}
		}
	}
};

/** Sphere Constraint functor that models an impassable sphere.
 */
struct SphereConstraint {
	double radius_;
	Point sphereCenter_;

	/** SphereConstraint Constructor.
	 * @param[in] radius Sets the radius of the sphere constraint.
	 * @param[in] sphereCenter Sets the center of the sphere constraint.
	 */
	SphereConstraint(double radius, Point sphereCenter) :
		radius_(radius), sphereCenter_(sphereCenter) {}

	/** Sphere Constraint Setter
	 * @param[in] g Valid graph.
	 * @param[in] t Valid time.
	 * @post Set nodes that moves into the sphere constraint to the nearest point on the sphere.
	 * @post Set the component of the node's velocity v = v - (v*R)*R where R = (x - sphereCenter)/distance(x - sphereCenter)
	 */
	void operator()(GraphType& g, double t) {
		(void) t;     // silence compiler warnings
		for(auto it = g.node_begin(); it != g.node_end(); ++it) {
			Node n = (*it);
			double distance = norm(n.position() - sphereCenter_);
			//Set nodes to be on the sphere
			if( distance < radius_ ){
				n.position() = (n.position() - sphereCenter_) * (radius_ / distance) + sphereCenter_;
				Point R = (n.position() - sphereCenter_)/distance;
				n.value().velocity -= dot(n.value().velocity, R) * R;
			}
		}
	}
};

/** Remove sphere functor that removes nodes that touch a sphere constraint.
 *  are removed.
 */
struct RemoveSphere {
	double radius_;
	Point sphereCenter_;

	/** RemoveSphere Constructor.
	 * @param[in] radius Sets the radius of the remove sphere area.
	 * @param[in] sphereCenter Sets the center of the remove sphere area.
	 */
	RemoveSphere(double radius, Point sphereCenter) :
		radius_(radius), sphereCenter_(sphereCenter) {}

	/** Remove Sphere Setter
	 * @param[in] g Valid graph.
	 * @param[in] t Valid time.
	 * @post nodes that move through the sphere's volume are removed from the graph.
	 * @post new @ g.num_nodes() = old @ g.num_nodes() - number of removed nodes.
	 * @post new @ g.num_edges() = old @ g.num_edges() - number of removed nodes * each nodes' degree().
	 */
	void operator()(GraphType& g, double t) {
		(void) t;     // silence compiler warnings
		for(auto it = g.node_begin(); it != g.node_end(); ++it) {
			Node n = (*it);
			double distance = norm(n.position() - sphereCenter_);
			//Set nodes to be on the sphere
			if( distance < radius_ ){
				g.remove_node(n);
			}
		}
	}
};

/** Combine Constraints Functor that returns a combination of constraints
 * @param[in] Two valid constraints in c1 and c2.
 */
template <typename Constraint1, typename Constraint2>
struct CombineConstraints {
	Constraint1 c1_;
	Constraint2 c2_;

	CombineConstraints(Constraint1 c1, Constraint2 c2)
			: c1_(c1), c2_(c2) {}

	void operator() (GraphType& g, double t){
		c1_(g,t);
		c2_(g,t);
	}
};

/** Combine Constraints Function that returns a combination of constraints
 * @param[in] Two valid constraints in @a c1, and @a c2.
 * @pre Valid constraints that input a node and time with a function call of c(g,t).
 * @return A CombineConstraints object that executes the constraints.
 */
template <typename Constraint1, typename Constraint2>
CombineConstraints<Constraint1, Constraint2> make_combined_constraints(Constraint1 c1, Constraint2 c2){
	return CombineConstraints<Constraint1, Constraint2> ({c1, c2});
}

/** Combine Constraints Function that returns a combination of constraints
 * @param[in] Valid constraints in @a c1, @a c2, and @a c3.
 * @pre Valid constraints that input a node and time with a function call of c(g,t).
 * @return A CombineConstraints object that executes the constraints.
 */
template <typename Constraint1, typename Constraint2, typename Constraint3>
CombineConstraints<CombineConstraints<Constraint1, Constraint2>, Constraint3> make_combined_constraints(Constraint1 c1, Constraint2 c2, Constraint3 c3){
	return CombineConstraints<CombineConstraints<Constraint1, Constraint2>, Constraint3> ({{c1, c2}, c3});
}

int main(int argc, char** argv) {
	// Check arguments
	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " NODES_FILE TETS_FILE\n";
		exit(1);
	}

	// Construct a graph
	GraphType graph;

	// Create a nodes_file from the first input argument
	std::ifstream nodes_file(argv[1]);
	// Interpret each line of the nodes_file as a 3D Point and add to the Graph
	std::vector<Node> nodes;
	Point p;
	while (CS207::getline_parsed(nodes_file, p))
		nodes.push_back(graph.add_node(p));

	// Create a tets_file from the second input argument
	std::ifstream tets_file(argv[2]);
	// Interpret each line of the tets_file as four ints which refer to nodes
	std::array<int,4> t;
	while (CS207::getline_parsed(tets_file, t)) {
		for (unsigned i = 1; i < t.size(); ++i) {
			graph.add_edge(nodes[t[0]], nodes[t[1]]);
			graph.add_edge(nodes[t[0]], nodes[t[2]]);
#if 1
			// Diagonal edges:
			graph.add_edge(nodes[t[0]], nodes[t[3]]);
			graph.add_edge(nodes[t[1]], nodes[t[2]]);
#endif
			graph.add_edge(nodes[t[1]], nodes[t[3]]);
			graph.add_edge(nodes[t[2]], nodes[t[3]]);
		}
	}

	// Set initial conditions for your nodes, if necessary.
	// Construct Forces/Constraints

	//Zero initial velocity
	//Set mass
	for (auto it = graph.node_begin(); it != graph.node_end(); ++it) {
		auto n = *it;
		n.value().velocity = Point(0,0,0);
		n.value().mass = 1.0/graph.num_nodes();
	}

	//To set initial_length for Problem1Force
	for (GraphType::EdgeIterator ei = graph.edge_begin(); ei != graph.edge_end(); ++ei ) {
		Initial_length = (*ei).length();
	}

	//To set rest length for all of the Edges to their initial length
	for (GraphType::EdgeIterator ei = graph.edge_begin(); ei != graph.edge_end(); ++ei ) {
		(*ei).value().length = (*ei).length();
	}

	// Print out the stats
	std::cout << graph.num_nodes() << " " << graph.num_edges() << std::endl;

	// Launch the SDLViewer
	CS207::SDLViewer viewer;
	auto node_map = viewer.empty_node_map(graph);
	viewer.launch();

	viewer.add_nodes(graph.node_begin(), graph.node_end(), node_map);
	viewer.add_edges(graph.edge_begin(), graph.edge_end(), node_map);

	viewer.center_view();

	// Begin the mass-spring simulation
	double dt = 0.001;
	double t_start = 0.0;
	double t_end   = 5.0;

	for (double t = t_start; t < t_end; t += dt) {
		//std::cout << "t = " << t << std::endl;

		//Setting Forces
		GravityForce gravity(grav);
		MassSpringForce massSpringForce(100);
		DampingForce damping(1.0/graph.num_nodes());
		//Final Force
		auto f = make_combined_force( gravity, massSpringForce, damping);

		//Setting Constraints
		HPlaneConstraint hplane(-0.75);
		SphereConstraint sphereConstraint(0.15, Point(0.5, 0.5, -0.5));
		RemoveSphere removeSphere(0.15, Point(0.5, 0.5, -0.5));

		//Final Constraints
		//Uncomment to run different constraints
		//auto c = make_combined_constraints(hplane, FixedConstraint()); //Problem 4 Part 1
		//auto c = make_combined_constraints(sphereConstraint, FixedConstraint()); //Problem 4 Part 2
		//auto c = make_combined_constraints(removeSphere, FixedConstraint()); //Problem 4 Part 3
		//auto c = make_combined_constraints( hplane, removeSphere, FixedConstraint()); //Mix of constraints
		auto c = make_combined_constraints( hplane, sphereConstraint, FixedConstraint()); //Mix of constraints

		//Set symp_euler_step
		//Problem 1 force
		//symp_euler_step(graph, t, dt, Problem1Force(), FixedConstraint()); //Problem 1

		//Just force ;No constraints
		//symp_euler_step(graph, t, dt, f); //Problem 3

		//Final Version - Add constraints
		symp_euler_step(graph, t, dt, f, c);

		//For remove
		viewer.clear();
		node_map.clear();
		viewer.add_nodes(graph.node_begin(), graph.node_end(), node_map);
		viewer.add_edges(graph.edge_begin(), graph.edge_end(), node_map);

		// Update viewer with nodes' new positions
		viewer.add_nodes(graph.node_begin(), graph.node_end(), node_map);
		viewer.set_label(t);

		// These lines slow down the animation for small graphs, like grid0_*.
		// Feel free to remove them or tweak the constants.
		if (graph.size() < 100)
			CS207::sleep(0.001);
	}

	return 0;
}
