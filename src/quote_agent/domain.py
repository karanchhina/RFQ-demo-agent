"""Small business domain for the quote-agent demo."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ValidationError

DISTRIBUTOR = {"id": "acme", "name": "Acme Industrial Supply"}

ACCOUNTS = {
    "nova": {
        "account_id": "nova",
        "name": "Nova Robotics",
        "default_ship_to": "Detroit, MI",
        "payment_terms": "Net 30",
    },
    "peak": {
        "account_id": "peak",
        "name": "Peak Manufacturing",
        "default_ship_to": "Austin, TX",
        "payment_terms": "Net 45",
    },
}

PRODUCTS = {
    "VAL-SS-100-FLOWFORGE": {
        "sku": "VAL-SS-100-FLOWFORGE",
        "name": "1-inch stainless ball valve",
        "brand": "FlowForge",
        "category": "valve",
        "aliases": ["one inch stainless valve", "FlowForge V100"],
        "substitutes": ["VAL-SS-100-BLUEPEAK", "VAL-SS-100-NORTHSTAR"],
    },
    "VAL-SS-150-FLOWFORGE": {
        "sku": "VAL-SS-150-FLOWFORGE",
        "name": "1.5-inch stainless ball valve",
        "brand": "FlowForge",
        "category": "valve",
        "aliases": ["one and a half inch stainless valve", "FlowForge V150"],
        "substitutes": [],
    },
    "VAL-SS-100-BLUEPEAK": {
        "sku": "VAL-SS-100-BLUEPEAK",
        "name": "1-inch stainless ball valve",
        "brand": "BluePeak",
        "category": "valve",
        "aliases": ["BluePeak stainless valve", "BluePeak V100"],
        "substitutes": ["VAL-SS-100-FLOWFORGE", "VAL-SS-100-NORTHSTAR"],
    },
    "VAL-SS-100-NORTHSTAR": {
        "sku": "VAL-SS-100-NORTHSTAR",
        "name": "1-inch stainless ball valve",
        "brand": "Northstar",
        "category": "valve",
        "aliases": ["Northstar stainless valve", "Northstar V100"],
        "substitutes": ["VAL-SS-100-FLOWFORGE", "VAL-SS-100-BLUEPEAK"],
    },
    "BRG-6205-AXIS": {
        "sku": "BRG-6205-AXIS",
        "name": "6205 sealed ball bearing",
        "brand": "Axis",
        "category": "bearing",
        "aliases": ["6205 bearing", "25x52x15 bearing"],
        "substitutes": [],
    },
    "PMP-100-RIVET": {
        "sku": "PMP-100-RIVET",
        "name": "1-inch centrifugal pump",
        "brand": "Rivet",
        "category": "pump",
        "aliases": ["Rivet P100", "one inch centrifugal pump"],
        "substitutes": ["PMP-100-CASCADE"],
    },
    "PMP-100-CASCADE": {
        "sku": "PMP-100-CASCADE",
        "name": "1-inch centrifugal pump",
        "brand": "Cascade",
        "category": "pump",
        "aliases": ["Cascade P100"],
        "substitutes": ["PMP-100-RIVET"],
    },
}

PRICING_INVENTORY = {
    "VAL-SS-100-FLOWFORGE": {
        "list_price": 145.00,
        "unit_cost": 92.00,
        "stock": 0,
        "restock_days": 21,
    },
    "VAL-SS-150-FLOWFORGE": {
        "list_price": 189.00,
        "unit_cost": 121.00,
        "stock": 18,
        "restock_days": 14,
    },
    "VAL-SS-100-BLUEPEAK": {
        "list_price": 139.00,
        "unit_cost": 88.00,
        "stock": 65,
        "restock_days": 10,
    },
    "VAL-SS-100-NORTHSTAR": {
        "list_price": 152.00,
        "unit_cost": 96.00,
        "stock": 40,
        "restock_days": 12,
    },
    "BRG-6205-AXIS": {
        "list_price": 18.00,
        "unit_cost": 10.00,
        "stock": 250,
        "restock_days": 5,
    },
    "PMP-100-RIVET": {
        "list_price": 425.00,
        "unit_cost": 280.00,
        "stock": 0,
        "restock_days": 30,
    },
    "PMP-100-CASCADE": {
        "list_price": 438.00,
        "unit_cost": 290.00,
        "stock": 24,
        "restock_days": 12,
    },
}

ACCOUNT_DISCOUNTS = {"nova": 0.05, "peak": 0.00}
APPROVAL_POLICIES = {"peak": {"cross_brand_substitute": "sales_manager"}}


class AccountMemory(BaseModel):
    preferred_skus: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)


class MemoryDecision(BaseModel):
    save: bool = Field(
        description="Whether the human response contains a reusable account preference"
    )
    preferred_skus: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(
        default_factory=list,
        description="Complete revised list of reusable account preferences",
    )


INITIAL_DISTRIBUTOR_MEMORY = [
    "Approved valve-substitute brands are FlowForge, BluePeak, and Northstar.",
    "Only propose SKUs listed as substitutes for the requested product.",
    "When a requested product is unavailable from stock, evaluate its listed substitutes before drafting a backorder quote.",
    "Never silently replace a requested SKU; clearly identify any proposed substitute.",
    "When several compatible approved substitutes are in stock and no account preference applies, prefer the lowest-priced option.",
]

INITIAL_ACCOUNT_MEMORIES = {
    "nova": AccountMemory(
        preferred_skus=["BRG-6205-AXIS"],
        preferences=[
            "Nova Robotics prefers Axis bearings in manufacturer-sealed packaging."
        ],
    ),
    "peak": AccountMemory(
        preferred_skus=["VAL-SS-100-BLUEPEAK"],
        preferences=[
            "Peak Manufacturing accepts BluePeak for this FlowForge valve."
        ],
    ),
}

ACCOUNT_MEMORY_KEY = "profile"


def memory_namespace(path: str) -> tuple[str, ...]:
    return tuple(part for part in path.split("/") if part)


def account_memory(account_id: str) -> tuple[str, ...]:
    return memory_namespace(
        f"/distributor/{DISTRIBUTOR['id']}/account/{account_id}/memory/"
    )


DISTRIBUTOR_MEMORY = memory_namespace(
    f"/distributor/{DISTRIBUTOR['id']}/memory/"
)


def _account_memory_from_value(value: dict) -> AccountMemory | None:
    if isinstance(value.get("preferences"), list):
        try:
            return AccountMemory.model_validate(value)
        except ValidationError:
            return None
    text = value.get("text")
    preferred_skus = value.get("preferred_skus", [])
    if not isinstance(text, str) or not isinstance(preferred_skus, list):
        return None
    return AccountMemory(
        preferred_skus=preferred_skus,
        preferences=[text],
    )


def _merge_account_memories(memories: list[AccountMemory]) -> AccountMemory:
    preferred_skus = []
    preferences = []
    seen_preferences = set()
    for memory in memories:
        for sku in memory.preferred_skus:
            normalized_sku = sku.strip().upper()
            if normalized_sku in PRODUCTS and normalized_sku not in preferred_skus:
                preferred_skus.append(normalized_sku)
        for preference in memory.preferences:
            normalized_preference = " ".join(preference.split())
            identity = normalized_preference.casefold()
            if normalized_preference and identity not in seen_preferences:
                preferences.append(normalized_preference)
                seen_preferences.add(identity)
    return AccountMemory(
        preferred_skus=preferred_skus,
        preferences=preferences,
    )


def _consolidated_account_memory(
    initial: AccountMemory, items
) -> AccountMemory:
    items = list(items)
    profile_item = next(
        (item for item in items if item.key == ACCOUNT_MEMORY_KEY), None
    )
    profile = (
        _account_memory_from_value(profile_item.value) if profile_item else None
    )
    memories = [profile] if profile else [initial]
    memories.extend(
        memory
        for item in items
        if item.key != ACCOUNT_MEMORY_KEY
        if (memory := _account_memory_from_value(item.value)) is not None
    )
    return _merge_account_memories(memories)


def seed_memories(store) -> None:
    """Seed missing records without overwriting learned account memory."""
    for index, text in enumerate(INITIAL_DISTRIBUTOR_MEMORY, start=1):
        key = f"seed-{index}"
        if store.get(DISTRIBUTOR_MEMORY, key) is None:
            store.put(DISTRIBUTOR_MEMORY, key, {"text": text})
    for account_id, memory in INITIAL_ACCOUNT_MEMORIES.items():
        namespace = account_memory(account_id)
        items = list(store.search(namespace))
        profile = _consolidated_account_memory(memory, items)
        store.put(namespace, ACCOUNT_MEMORY_KEY, profile.model_dump())
        for item in items:
            if item.key != ACCOUNT_MEMORY_KEY:
                store.delete(namespace, item.key)


async def aseed_memories(store) -> None:
    """Async Agent Server equivalent of ``seed_memories``."""
    for index, text in enumerate(INITIAL_DISTRIBUTOR_MEMORY, start=1):
        key = f"seed-{index}"
        if await store.aget(DISTRIBUTOR_MEMORY, key) is None:
            await store.aput(DISTRIBUTOR_MEMORY, key, {"text": text})
    for account_id, memory in INITIAL_ACCOUNT_MEMORIES.items():
        namespace = account_memory(account_id)
        items = list(await store.asearch(namespace))
        profile = _consolidated_account_memory(memory, items)
        await store.aput(namespace, ACCOUNT_MEMORY_KEY, profile.model_dump())
        for item in items:
            if item.key != ACCOUNT_MEMORY_KEY:
                await store.adelete(namespace, item.key)


def read_memories(store, namespace: tuple[str, ...]) -> list[str]:
    memories: list[str] = []
    seen: set[str] = set()
    for item in store.search(namespace):
        text = item.value["text"]
        normalized = " ".join(text.casefold().split())
        if normalized not in seen:
            memories.append(text)
            seen.add(normalized)
    return memories


async def aread_memories(store, namespace: tuple[str, ...]) -> list[str]:
    memories: list[str] = []
    seen: set[str] = set()
    for item in await store.asearch(namespace):
        text = item.value["text"]
        normalized = " ".join(text.casefold().split())
        if normalized not in seen:
            memories.append(text)
            seen.add(normalized)
    return memories


def read_account_memory(store, account_id: str) -> AccountMemory:
    item = store.get(account_memory(account_id), ACCOUNT_MEMORY_KEY)
    if item is None:
        return AccountMemory()
    return AccountMemory.model_validate(item.value)


async def aread_account_memory(store, account_id: str) -> AccountMemory:
    item = await store.aget(account_memory(account_id), ACCOUNT_MEMORY_KEY)
    if item is None:
        return AccountMemory()
    return AccountMemory.model_validate(item.value)


def preferred_account_skus(memory: AccountMemory) -> list[str]:
    return memory.preferred_skus


def memory_from_decision(decision: MemoryDecision) -> AccountMemory | None:
    if not decision.save:
        return None
    memory = _merge_account_memories([AccountMemory(
        preferred_skus=decision.preferred_skus,
        preferences=decision.preferences,
    )])
    if not memory.preferences:
        raise ValueError("Saved account memory requires at least one preference")
    return memory


def apply_memory_decision(
    store, account_id: str, decision: MemoryDecision
) -> bool:
    memory = memory_from_decision(decision)
    if memory is None:
        return False
    if read_account_memory(store, account_id) == memory:
        return False
    store.put(
        account_memory(account_id), ACCOUNT_MEMORY_KEY, memory.model_dump()
    )
    return True


async def aapply_memory_decision(
    store, account_id: str, decision: MemoryDecision
) -> bool:
    memory = memory_from_decision(decision)
    if memory is None:
        return False
    if await aread_account_memory(store, account_id) == memory:
        return False
    await store.aput(
        account_memory(account_id), ACCOUNT_MEMORY_KEY, memory.model_dump()
    )
    return True


def check_price_and_stock(account_id: str, sku: str, quantity: int) -> dict:
    if quantity <= 0:
        return {
            "status": "invalid_request",
            "message": "quantity must be greater than zero",
        }

    normalized_account_id = account_id.strip().lower()
    normalized_sku = sku.strip().upper()
    if normalized_account_id not in ACCOUNTS:
        return {
            "status": "not_found",
            "entity": "account",
            "value": normalized_account_id,
        }
    if normalized_sku not in PRODUCTS:
        return {
            "status": "not_found",
            "entity": "product",
            "value": normalized_sku,
        }

    facts = PRICING_INVENTORY[normalized_sku]
    discount = ACCOUNT_DISCOUNTS[normalized_account_id]
    unit_price = round(facts["list_price"] * (1 - discount), 2)
    inventory_quantity = facts["stock"]
    in_stock = inventory_quantity >= quantity
    return {
        "status": "ok",
        "account_id": normalized_account_id,
        "sku": normalized_sku,
        "quantity": quantity,
        "unit_price": unit_price,
        "extended_price": round(unit_price * quantity, 2),
        "unit_cost": facts["unit_cost"],
        "margin_pct": round(
            (unit_price - facts["unit_cost"]) / unit_price * 100, 2
        ),
        "inventory_quantity": inventory_quantity,
        "availability": "in_stock" if in_stock else "backorder",
        "lead_time_days": 0 if in_stock else facts["restock_days"],
    }


@dataclass
class MockERP:
    """Process-local stand-in for a quote-writing ERP boundary."""

    created_quotes: list[dict] = field(default_factory=list)
    by_execution: dict[str, dict] = field(default_factory=dict)

    def clear(self) -> None:
        self.created_quotes.clear()
        self.by_execution.clear()
