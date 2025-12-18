# Example: Ancestor Relations (Section 2.1)
# This demonstrates the core insight: Datalog rules are einsums over
# Boolean tensors with a step function.
# Ancestor(x,z) <- Ancestor(x,y), Parent(y,z) == Axz = H(Σy Axy * Pyz)
# Where H is the Heaviside step function.

Parent(Alice, Bob)
Parent(Bob, Charlie)
Parent(Charlie, David)
Parent(Eve, Frank)
Parent(Frank, Grace)

Ancestor(x, y) <- Parent(x, y)
Ancestor(x, z) <- Ancestor(x, y), Parent(y, z)

# After forward chaining, query:
# ?Ancestor(Alice, x)   → Bob, Charlie, David
# ?Ancestor(x, David)   → Alice, Bob, Charlie
# ?Ancestor(x, y)       → all ancestor pairs
