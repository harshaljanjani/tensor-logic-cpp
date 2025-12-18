# Knowledge Representation (Section 5, Reasoning in Embedding Space)
# This demonstrates the symbolic reasoning that can be combined
# with embeddings for analogical inference.

ISA(Dog, Mammal)
ISA(Cat, Mammal)
ISA(Mammal, Animal)
ISA(Animal, LivingThing)
ISA(Bird, Animal)
ISA(Penguin, Bird)

ISA(x, z) <- ISA(x, y), ISA(y, z)

HasPart(Mammal, Heart)
HasPart(Bird, Wings)
HasPart(Bird, Beak)

HasPart(x, z) <- ISA(x, y), HasPart(y, z)

CanDo(Bird, Fly)
CanDo(Mammal, Breathe)
CannotDo(Penguin, Fly)

CanDo(x, a) <- ISA(x, y), CanDo(y, a)

# After forward chaining:
# ?ISA(Dog, x)     → Mammal, Animal, LivingThing
# ?HasPart(Dog, x) → Heart
# ?CanDo(Cat, x)   → Breathe
