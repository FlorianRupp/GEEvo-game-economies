graph [
  directed 1
  node [
    id 0
    label "Townhall Level"
    type "pool"
    value "1"
  ]
  node [
    id 1
    label "TH Gold & Elixir Capacity"
    type "register"
    value "subset(list,index(lvl))"
  ]
  node [
    id 2
    label "Capacity list (TH)"
    type "list"
    value "[1K,2.5K,10K,50K...]"
  ]
  node [
    id 3
    label "Player total gold capacity"
    type "register"
    value "townHall + storage"
  ]
  node [
    id 4
    label "Inventory control (fx)"
    type "register"
    value "inventory-capacity"
  ]
  node [
    id 5
    label "Gate (gold inv input)"
    type "gate"
    value ""
  ]
  node [
    id 6
    label "Gold inventory"
    type "pool"
    value "950"
  ]
  node [
    id 7
    label "Drain (gold inventory)"
    type "drain"
    value ""
  ]
  node [
    id 8
    label "Gate (Gold Storage upgrade)"
    type "gate"
    value ""
  ]
  node [
    id 9
    label "Converter (Gold Storage upgrade)"
    type "converter"
    value ""
  ]
  node [
    id 10
    label "Gold Storage level"
    type "pool"
    value "0"
  ]
  node [
    id 11
    label "Storage Capacity"
    type "register"
    value "subset(list,index(lvl))"
  ]
  node [
    id 12
    label "Capacity list (GS)"
    type "list"
    value "[1.5K,3K,6K,12K...]"
  ]
  node [
    id 13
    label "Build duration (GS)"
    type "register"
    value "subset(list,index(lvl))"
  ]
  node [
    id 14
    label "Build duration list (GS)"
    type "list"
    value "[1,5,20,60,120...]"
  ]
  node [
    id 15
    label "Upgrade cost (GS)"
    type "register"
    value "subset(list,index(lvl))"
  ]
  node [
    id 16
    label "Cost list (GS)"
    type "list"
    value "[300,750,1.5K,3K...]"
  ]
  node [
    id 17
    label "Gate (Gold Mine upgrade)"
    type "gate"
    value ""
  ]
  node [
    id 18
    label "Converter (Gold Mine upgrade)"
    type "converter"
    value ""
  ]
  node [
    id 19
    label "Gold mine level"
    type "pool"
    value "1"
  ]
  node [
    id 20
    label "Capacity (GM)"
    type "register"
    value "subset(list,index(lvl))"
  ]
  node [
    id 21
    label "Production"
    type "register"
    value "subset(list,index(lvl))"
  ]
  node [
    id 22
    label "Production list (per min)"
    type "list"
    value "[3,7,10,13,17,2...]"
  ]
  node [
    id 23
    label "Capacity list (GM)"
    type "list"
    value "[1K,2K,3K,5K,10K...]"
  ]
  node [
    id 24
    label "Build duration (GM)"
    type "register"
    value "subset(list,index(lvl))"
  ]
  node [
    id 25
    label "Build duration list (GM)"
    type "list"
    value "[1,1,4,10,40,18...]"
  ]
  node [
    id 26
    label "Upgrade cost (GM)"
    type "register"
    value "subset(list,index(lvl))"
  ]
  node [
    id 27
    label "Cost list (GM)"
    type "list"
    value "[150,300,700,1.4K...]"
  ]
  node [
    id 28
    label "Gold Mine (source)"
    type "source"
    value ""
  ]
  node [
    id 29
    label "Gold mine storage"
    type "pool"
    value "20K cap"
  ]
  node [
    id 30
    label "Collect Gold (gate)"
    type "gate"
    value "tiap 30 mnt"
  ]
  node [
    id 31
    label "Elixir inventory"
    type "pool"
    value "200"
  ]
  edge [
    source 2
    target 1
    label "list"
    type "state"
  ]
  edge [
    source 0
    target 1
    label "townHall (lvl)"
    type "state"
  ]
  edge [
    source 1
    target 3
    label "storage"
    type "state"
  ]
  edge [
    source 0
    target 3
    label "townHall"
    type "state"
  ]
  edge [
    source 3
    target 4
    label "capacity"
    type "state"
  ]
  edge [
    source 4
    target 5
    label "inventory-capacity"
    type "state"
  ]
  edge [
    source 5
    target 6
    label "resource flow"
    type "resource"
  ]
  edge [
    source 6
    target 7
    label "drain"
    type "resource"
  ]
  edge [
    source 6
    target 8
    label "upgrade cost (GS)"
    type "resource"
  ]
  edge [
    source 6
    target 17
    label "upgrade cost (GM)"
    type "resource"
  ]
  edge [
    source 8
    target 9
    label "trigger"
    type "resource"
  ]
  edge [
    source 9
    target 10
    label "+1 level"
    type "resource"
  ]
  edge [
    source 10
    target 11
    label "lvl"
    type "state"
  ]
  edge [
    source 11
    target 3
    label "storage capacity"
    type "state"
  ]
  edge [
    source 12
    target 11
    label "list"
    type "state"
  ]
  edge [
    source 14
    target 13
    label "list"
    type "state"
  ]
  edge [
    source 13
    target 9
    label "build timer"
    type "state"
  ]
  edge [
    source 15
    target 8
    label "upgrade cost"
    type "state"
  ]
  edge [
    source 16
    target 15
    label "list"
    type "state"
  ]
  edge [
    source 17
    target 18
    label "trigger"
    type "resource"
  ]
  edge [
    source 18
    target 19
    label "+1 level"
    type "resource"
  ]
  edge [
    source 19
    target 20
    label "lvl"
    type "state"
  ]
  edge [
    source 19
    target 21
    label "lvl"
    type "state"
  ]
  edge [
    source 22
    target 21
    label "list"
    type "state"
  ]
  edge [
    source 23
    target 20
    label "list"
    type "state"
  ]
  edge [
    source 20
    target 29
    label "caps storage"
    type "state"
  ]
  edge [
    source 21
    target 28
    label "production rate"
    type "state"
  ]
  edge [
    source 25
    target 24
    label "list"
    type "state"
  ]
  edge [
    source 24
    target 18
    label "build timer"
    type "state"
  ]
  edge [
    source 26
    target 17
    label "upgrade cost"
    type "state"
  ]
  edge [
    source 27
    target 26
    label "list"
    type "state"
  ]
  edge [
    source 28
    target 29
    label "produces gold"
    type "resource"
  ]
  edge [
    source 29
    target 30
    label "trigger (30 min)"
    type "resource"
  ]
  edge [
    source 30
    target 6
    label "collect gold"
    type "resource"
  ]
  edge [
    source 6
    target 31
    label "elixir flow"
    type "resource"
  ]
]
