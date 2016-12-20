package de.bückemeyer.backpropagation;


import de.bückemeyer.backpropagation.node.Node;

public class Connection {
    private double weight;
    private double oldWeight;
    private Node left;
    private Node right;

    public Connection(double weight, Node left, Node right) {
        this.weight = weight;
        this.left = left;
        this.right = right;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        oldWeight = this.weight;
        this.weight = weight;
    }

    public Node getLeft() {
        return left;
    }

    public void setLeft(Node left) {
        this.left = left;
    }

    public Node getRight() {
        return right;
    }

    public void setRight(Node right) {
        this.right = right;
    }

    public double getOldWeight() {
        return oldWeight;
    }
}
