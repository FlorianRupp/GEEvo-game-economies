graph [
  directed 1
  node [
    id 0
    label "Townhall Level"
    type "pool"
    value "1"
    geevo_type "Pool"
  ]
  node [
    id 1
    label "TH Gold &#38; Elixir Capacity"
    type "register"
    value "subset(list,index(lvl))"
    geevo_type "Pool"
  ]
  node [
    id 2
    label "Player total gold capacity"
    type "register"
    value "townHall + storage"
    geevo_type "Pool"
  ]
  node [
    id 3
    label "Inventory control (fx)"
    type "register"
    value "inventory-capacity"
    geevo_type "Pool"
  ]
  node [
    id 4
    label "Gate (gold inv input)"
    type "gate"
    value ""
    geevo_type "RandomGate"
  ]
  node [
    id 5
    label "Gold inventory"
    type "pool"
    value "950"
    geevo_type "Pool"
  ]
  node [
    id 6
    label "Drain (gold inventory)"
    type "drain"
    value ""
    geevo_type "RandomGate"
  ]
  node [
    id 7
    label "Gate (Gold Storage upgrade)"
    type "gate"
    value ""
    geevo_type "RandomGate"
  ]
  node [
    id 8
    label "Converter (Gold Storage upgrade)"
    type "converter"
    value ""
    geevo_type "Pool"
  ]
  node [
    id 9
    label "Gold Storage level"
    type "pool"
    value "0"
    geevo_type "Pool"
  ]
  node [
    id 10
    label "Storage Capacity"
    type "register"
    value "subset(list,index(lvl))"
    geevo_type "Pool"
  ]
  node [
    id 11
    label "Gate (Gold Mine upgrade)"
    type "gate"
    value ""
    geevo_type "Source"
  ]
  node [
    id 12
    label "Converter (Gold Mine upgrade)"
    type "converter"
    value ""
    geevo_type "Source"
  ]
  node [
    id 13
    label "Gold mine level"
    type "pool"
    value "1"
    geevo_type "Source"
  ]
  node [
    id 14
    label "Capacity (GM)"
    type "register"
    value "subset(list,index(lvl))"
    geevo_type "Pool"
  ]
  node [
    id 15
    label "Production"
    type "register"
    value "subset(list,index(lvl))"
    geevo_type "Converter"
  ]
  node [
    id 16
    label "Gold Mine (source)"
    type "source"
    value ""
    geevo_type "Source"
  ]
  node [
    id 17
    label "Gold mine storage"
    type "pool"
    value "20K cap"
    geevo_type "Source"
  ]
  node [
    id 18
    label "Collect Gold (gate)"
    type "gate"
    value "tiap 30 mnt"
    geevo_type "RandomGate"
  ]
  node [
    id 19
    label "Elixir inventory"
    type "pool"
    value "200"
    geevo_type "Pool"
  ]
  edge [
    source 0
    target 1
    label "townHall (lvl)"
    type "state"
  ]
  edge [
    source 0
    target 2
    label "townHall"
    type "state"
  ]
  edge [
    source 1
    target 2
    label "storage"
    type "state"
  ]
  edge [
    source 3
    target 4
    label "inventory-capacity"
    type "state"
  ]
  edge [
    source 4
    target 5
    label "resource flow"
    type "resource"
  ]
  edge [
    source 5
    target 6
    label "drain"
    type "resource"
  ]
  edge [
    source 5
    target 7
    label "upgrade cost (GS)"
    type "resource"
  ]
  edge [
    source 5
    target 19
    label "elixir flow"
    type "resource"
  ]
  edge [
    source 7
    target 8
    label "trigger"
    type "resource"
  ]
  edge [
    source 8
    target 9
    label "+1 level"
    type "resource"
  ]
  edge [
    source 9
    target 10
    label "lvl"
    type "state"
  ]
  edge [
    source 10
    target 2
    label "storage capacity"
    type "state"
  ]
  edge [
    source 11
    target 12
    label "trigger"
    type "resource"
  ]
  edge [
    source 12
    target 13
    label "+1 level"
    type "resource"
  ]
  edge [
    source 13
    target 14
    label "lvl"
    type "state"
  ]
  edge [
    source 13
    target 15
    label "lvl"
    type "state"
  ]
  edge [
    source 14
    target 17
    label "caps storage"
    type "state"
  ]
  edge [
    source 15
    target 16
    label "production rate"
    type "state"
  ]
  edge [
    source 16
    target 17
    label "produces gold"
    type "resource"
  ]
  edge [
    source 17
    target 18
    label "trigger (30 min)"
    type "resource"
  ]
  edge [
    source 18
    target 5
    label "collect gold"
    type "resource"
  ]
]
