"""
Base data models for GMTW-Ro task worlds
"""

from dataclasses import dataclass, field
from typing import Literal, Any
from enum import Enum


WorldType = Literal["travel", "schedule", "fact"]


class ConstraintType(str, Enum):
    """Type of constraint"""
    INSTRUCTION = "instruction"  # User-specified constraint from prompt
    STRUCTURAL = "structural"    # World physics/logic constraint


class GoalType(str, Enum):
    """Type of goal"""
    STRUCTURAL = "structural"  # Must be satisfied for valid solution
    OPTIONAL = "optional"      # Nice-to-have goals


@dataclass
class Constraint:
    """A constraint that must be checked"""
    id: str
    type: ConstraintType
    description_ro: str
    description_en: str = ""
    check_fn: str = ""  # Name of the function to call for verification
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Goal:
    """A goal to achieve in the world"""
    id: str
    type: GoalType
    description: str
    check_fn: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    """An entity in the world (attraction, meeting, fact, etc.)"""
    id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class World:
    """A complete task world specification"""
    world_id: str
    world_type: WorldType
    spec_version: str
    seed: int

    # Core world data
    payload: dict[str, Any]  # Actual world-specific data

    # Evaluation specification
    constraints: list[Constraint]
    goals: list[Goal]
    canonical_entities: dict[str, Entity]  # id -> Entity

    # Metadata
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "world_id": self.world_id,
            "world_type": self.world_type,
            "spec_version": self.spec_version,
            "seed": self.seed,
            "payload": self.payload,
            "constraints": [
                {
                    "id": c.id,
                    "type": c.type.value,
                    "description_ro": c.description_ro,
                    "description_en": c.description_en,
                    "check_fn": c.check_fn,
                    "params": c.params,
                }
                for c in self.constraints
            ],
            "goals": [
                {
                    "id": g.id,
                    "type": g.type.value,
                    "description": g.description,
                    "check_fn": g.check_fn,
                    "params": g.params,
                }
                for g in self.goals
            ],
            "canonical_entities": {
                eid: {
                    "id": e.id,
                    "name": e.name,
                    "aliases": e.aliases,
                    "attributes": e.attributes,
                }
                for eid, e in self.canonical_entities.items()
            },
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "World":
        """Create World from dictionary"""
        return cls(
            world_id=data["world_id"],
            world_type=data["world_type"],
            spec_version=data["spec_version"],
            seed=data["seed"],
            payload=data["payload"],
            constraints=[
                Constraint(
                    id=c["id"],
                    type=ConstraintType(c["type"]),
                    description_ro=c["description_ro"],
                    description_en=c.get("description_en", ""),
                    check_fn=c.get("check_fn", ""),
                    params=c.get("params", {}),
                )
                for c in data["constraints"]
            ],
            goals=[
                Goal(
                    id=g["id"],
                    type=GoalType(g["type"]),
                    description=g["description"],
                    check_fn=g["check_fn"],
                    params=g.get("params", {}),
                )
                for g in data["goals"]
            ],
            canonical_entities={
                eid: Entity(
                    id=e["id"],
                    name=e["name"],
                    aliases=e.get("aliases", []),
                    attributes=e.get("attributes", {}),
                )
                for eid, e in data["canonical_entities"].items()
            },
            meta=data.get("meta", {}),
        )


@dataclass
class Instance:
    """A complete evaluation instance (world + prompts)"""
    instance_id: str
    world: World
    prompt_ro: str
    prompt_en: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "instance_id": self.instance_id,
            "world": self.world.to_dict(),
            "prompt_ro": self.prompt_ro,
            "prompt_en": self.prompt_en,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Instance":
        """Create Instance from dictionary"""
        return cls(
            instance_id=data["instance_id"],
            world=World.from_dict(data["world"]),
            prompt_ro=data["prompt_ro"],
            prompt_en=data.get("prompt_en", ""),
            meta=data.get("meta", {}),
        )
