# R1Pro ↔ R1Lite Task Semantic Matching

Matching criteria (priority order):
1. Same core action + same/similar object
2. Same action category + related objects
3. Similar overall task semantics (robot actions, manipulation patterns)

Legend: **Bold** = strong match, normal = moderate match, *italic* = weak/partial match

| # | R1Pro Task | R1Lite Match 1 (Best) | R1Lite Match 2 | R1Lite Match 3 | Notes |
|---|---|---|---|---|---|
| 1 | turning_on_radio | **Turn_On_Off_The_Light** | **Switch_The_Fan_On_And_Off** | **Plug_And_Unplug_Switch_Lights** | All involve toggling electrical devices on/off |
| 2 | picking_up_trash | **Pick_Up_Trash_From_Ground** | **Pick_Up_Garbage_On_The_Ground** | **Dispose_Of_Garbage_In_The_Trash_Can** | Near-exact matches available |
| 3 | putting_away_Halloween_decorations | *Doll_Storage* | *Store_Dolls* | *Organize_Toys* | Weak: R1Lite has no seasonal decoration tasks; closest are storing decorative/toy items |
| 4 | cleaning_up_plates_and_food | **Handle_Plates** | **Dispose_Of_Leftover_Food** | **Store_Tableware** | Good matches for both plate handling and food disposal |
| 5 | can_meat | *Store_Fruit* | *Cooking_Food_In_An_Air_Fryer* | *Handle_The_Wooden_Barrel_Lid* | Weak: canning is a specialized food preservation task; R1Lite has no direct equivalent |
| 6 | setting_mousetraps | | | | No match: highly specialized mechanical task with no R1Lite equivalent |
| 7 | hiding_Easter_eggs | **Egg_Placement** | *Doll_Storage* | *Store_The_Dolls* | Egg_Placement is a strong match on object; others are weak on "hiding/placing small items" |
| 8 | picking_up_toys | **Organize_Toys** | **Store_Dolls** | **Store_The_Dolls** | Good matches: all involve collecting/organizing toy-like objects |
| 9 | rearranging_kitchen_furniture | **Chair_Push_And_Place** | **Push_And_Pull_The_Chair** | **Push_In_Chairs** | Good matches: all involve repositioning furniture |
| 10 | putting_up_Christmas_decorations_inside | *Flowers_Are_Inserted_Into_Vases* | *Arrange_Throw_Pillows_On_Living_Room_Sofa* | *Hang_The_Towel_On_The_Shelf* | Weak: R1Lite has no decoration-hanging tasks; closest are decorative arrangement actions |
| 11 | set_up_a_coffee_station_in_your_kitchen | *Make_Tea* | *Arrange_The_Tray* | *Organize_The_Condiment_Shelf* | Moderate: beverage prep + organizing kitchen station items |
| 12 | putting_dishes_away_after_cleaning | **Store_Tableware** | **Take_and_place_tableware** | **Handle_Plates** | Strong matches: all involve storing/placing clean tableware |
| 13 | preparing_lunch_box | *Make_Breakfast* | *Store_Fruit* | *Arrange_The_Tray* | Moderate: meal-prep and food arrangement tasks |
| 14 | loading_the_car | *From_Kitchen_To_Bedroom* | *Pushing_Garbage_Trucks_To_Put_Garbage_Away* | | Weak: R1Lite tasks are indoor only; no vehicle-loading equivalent |
| 15 | carrying_in_groceries | *From_Kitchen_To_Bedroom* | *Arrange_Mineral_Water_Bottles_On_The_Desk* | *Take_Drinks_From_The_Refrigerator_And_Place_Them_On_The_Table* | Weak: closest are item transport between rooms |
| 16 | bringing_in_wood | *From_Kitchen_To_Bedroom* | *Handle_The_Wooden_Barrel_Lid* | | Weak: outdoor heavy-carry task; no R1Lite equivalent |
| 17 | moving_boxes_to_storage | **Locker_torage_And_Retrieval** | **Put_The_Items_Into_The_Storage_Box** | *From_Kitchen_To_Bedroom* | Good on storage action; moderate on transport |
| 18 | bringing_water | **Pour_Water** | **Boil_The_Water** | **Water_Dispenser_Connected_To_Water** | Good matches: all involve water handling/transport |
| 19 | tidying_bedroom | **Make_The_Bed** | **Put_The_Pillow_On_The_Bed** | **Arrange_Sofa_Cushions** | Good matches: bedroom tidying and soft furnishing arrangement |
| 20 | outfit_a_basic_toolbox | **Storage_Tools** | *Pack_Medicines_Into_Medicine_Box* | *Pen_Case_For_Storing_Stationery* | Storage_Tools is a strong match; others are "organizing items into a container" |
| 21 | sorting_vegetables | **Arrange_Fruits** | **Arrange_The_Fruits** | **Store_Fruit** | Good: sorting/arranging produce items |
| 22 | collecting_childrens_toys | **Organize_Toys** | **Store_Dolls** | **Doll_Storage** | Strong matches: all involve collecting/organizing toys |
| 23 | putting_shoes_on_rack | **Taking_And_Placing_Shoes_In_The_Shoe_Cabinet** | **Put_Shoes_Into_The_Shoe_Box** | **Place_Slippers_In_The_Layered_Shoe_Cabinet** | Excellent matches: multiple shoe storage tasks |
| 24 | boxing_books_up_for_storage | **Put_The_Items_Into_The_Storage_Box** | **Locker_torage_And_Retrieval** | *Pen_Case_For_Storing_Stationery* | Good on boxing/storage action |
| 25 | storing_food | **Store_Fruit** | **Open_And_Close_The_Refrigerator_Drawer_To_Place_Items** | **Organize_Refrigerated_shelf** | Good: food storage and fridge organization |
| 26 | clearing_food_from_table_into_fridge | **Open_And_Close_The_Refrigerator_Drawer_To_Place_Items** | **Opening_Or_Closing_Refrigerator_Drawers_To_Store_And_Retrieve_Items** | **Dispose_Of_Leftover_Food** | Good: clearing table + fridge storage actions |
| 27 | assembling_gift_baskets | *Arrange_The_Tray* | *Arrange_Fruits* | *Organize_trays* | Moderate: arranging items on a tray/basket |
| 28 | sorting_household_items | **Organize_The_Toiletries** | **Tidy_Up_Bathroom_Toiletries** | **Arrange_Toiletries** | Good: all involve sorting/organizing household goods |
| 29 | getting_organized_for_work | **Put_The_Pen_Into_The_Pen_Holder** | **Pen_Case_For_Storing_Stationery** | *Open_And_Close_The_Notebook* | Good on stationery/work items organization |
| 30 | clean_up_your_desk | **Desktop_Garbage_Decluttering** | **Desktop_Trash_Disposal** | **Organizing_Desktop_Trash** | Excellent near-exact matches |
| 31 | setting_the_fire | | | | No match: fireplace/outdoor fire task with no R1Lite equivalent |
| 32 | clean_boxing_gloves | *Washing_Machine_Washes_Clothes* | *Fold_Clothes* | *Iron_The_Clothes* | Weak: closest are fabric/clothing maintenance tasks |
| 33 | wash_a_baseball_cap | **Washing_Machine_Washes_Clothes** | *Take_Out_The_Laundry_From_The_Washing_Machine* | *Clean_The_Sink* | Moderate: cap washing maps to laundry/washing tasks |
| 34 | wash_dog_toys | *Washing_Machine_Washes_Clothes* | *Clean_The_Sink* | *Organize_Toys* | Weak: combines washing + toy handling; no direct match |
| 35 | hanging_pictures | **Hang_The_Towel_On_The_Shelf** | **Organize_The_Wall-mounted_Coat_Rack** | *Hang_Clothes* | Moderate: all involve hanging items on wall/rack |
| 36 | attach_a_camera_to_a_tripod | *Connect_Router_Cables* | *Plug_Into_A_Fixed_Socket* | *Plug_The_Socket* | Weak: closest are device-connecting tasks |
| 37 | clean_a_patio | **Floor_cloth_wiping_stains** | **Wipe_The_Sewage_Stains_With_A_Ground_Cloth** | **Lift_up_the_carpet_and_wipe_the_floor** | Good: ground/floor cleaning and wiping actions |
| 38 | clean_a_trumpet | *Clean_The_Sink* | *Clean_The_Mirror* | *Clean_Cutting_Board* | Weak: generic cleaning tasks; no instrument-specific match |
| 39 | spraying_for_bugs | | | | No match: spraying/pest control has no R1Lite equivalent |
| 40 | spraying_fruit_trees | | | | No match: outdoor agricultural task with no R1Lite equivalent |
| 41 | make_microwave_popcorn | **Heat_Food_In_Microwave_Oven** | **Heat_Food_In_The_Microwave** | **Take_And_Put_Items_In_And_Out_Of_The_Microwave_Oven** | Excellent: microwave operation tasks |
| 42 | cook_cabbage | **Stir-Fry_Dishes** | **Stir-fry** | **Cooking_Food_In_An_Air_Fryer** | Good: stovetop/appliance cooking |
| 43 | chop_an_onion | *Stir-Fry_Dishes* | *Clean_The_Cutting_Board* | *Arrange_Fruits* | Weak: R1Lite has no cutting/chopping tasks; closest are cooking-adjacent |
| 44 | slicing_vegetables | *Stir-Fry_Dishes* | *Clean_The_Chopping_Board* | *Arrange_The_Fruits* | Weak: same as above, no slicing/cutting in R1Lite |
| 45 | chopping_wood | | | | No match: heavy outdoor labor task with no R1Lite equivalent |
| 46 | cook_hot_dogs | **Stir-Fry_Dishes** | **Cooking_Food_In_An_Air_Fryer** | **Heat_Food_In_Microwave_Oven** | Good: general cooking/heating tasks |
| 47 | cook_bacon | **Stir-frying** | **Stir-Fry_Dishes** | **Cooking_Food_In_An_Air_Fryer** | Good: pan/appliance cooking tasks |
| 48 | freeze_pies | **Open_and_close_the_freezer_door** | **Store_Fruit** | **Open_And_Close_The_Refrigerator_Drawer_To_Place_Items** | Good: freezer use + food storage |
| 49 | canning_food | *Store_Fruit* | *Handle_The_Wooden_Barrel_Lid* | *Organize_Refrigerator_Items* | Weak: food preservation has no direct match; closest are food storage |
| 50 | make_pizza | **Make_Breakfast** | **Cooking_Food_In_An_Air_Fryer** | **Stir-Fry_Dishes** | Good: meal preparation and cooking |

## Summary Statistics

| Match Quality | Count | Percentage |
|---|---|---|
| Has strong match (>=1 bold) | 35 | 70% |
| Moderate matches only | 7 | 14% |
| Weak matches only | 4 | 8% |
| No match (empty) | 4 | 8% |
| **Total** | **50** | **100%** |

### R1Pro Tasks With No R1Lite Equivalent
- **setting_mousetraps** (6): specialized mechanical device setup
- **setting_the_fire** (31): fireplace/campfire task
- **spraying_for_bugs** (39): pest control spraying
- **spraying_fruit_trees** (40): agricultural spraying
- **chopping_wood** (45): heavy outdoor labor

### Well-Matched Categories
- **Cleaning/wiping**: R1Lite has extensive floor, desk, toilet, sink cleaning tasks
- **Cooking**: R1Lite covers stir-frying, microwave, air fryer, steaming, boiling
- **Organizing/storing**: R1Lite excels here with shelf, cabinet, fridge, desk organization
- **Shoe storage**: R1Lite has 5+ shoe cabinet/box variants
- **Trash handling**: R1Lite has direct pick-up-trash and garbage disposal tasks

### Poorly-Matched Categories
- **Outdoor tasks**: R1Lite is entirely indoor; no yard, garden, or vehicle tasks
- **Food preparation** (cutting/chopping): R1Lite has cooking but no cutting/slicing
- **Seasonal/decorative**: No holiday decoration tasks in R1Lite
- **Specialized cleaning**: No instrument, sports equipment, or pet toy cleaning

# Selected

| # | R1Pro Task | R1Lite Match 1 (Best) | R1Lite Match 2 | R1Lite Match 3 |
|---|---|---|---|---|
| 1 | turning_on_radio | **Turn_On_Off_The_Light** | **Switch_The_Fan_On_And_Off** | **Plug_And_Unplug_Switch_Lights** |
| 17 | moving_boxes_to_storage | **Locker_torage_And_Retrieval** | **Put_The_Items_Into_The_Storage_Box** | *Put_The_Phone_Back_In_place* |
| 20 | outfit_a_basic_toolbox | **Storage_Tools** | *Take_And_Put_Items_In_And_Out_Of_The_Microwave_Oven* | *Iron_The_Clothes* |
| 24 | boxing_books_up_for_storage | **Put_The_Items_Into_The_Storage_Box** | **Locker_torage_And_Retrieval** | *Open_And_Close_The_Notebook* |
| 36 | attach_a_camera_to_a_tripod |

- Take_And_Place_The_Portable_Power_Bank20250628_002
- Put_The_Phone_Back_In_place_20250716_006
- Put_The_Items_Into_The_Storage_Box_20250929_002_007
- Storage_Tools_20250802_012
- Take_And_Put_Items_In_And_Out_Of_The_Microwave_Oven_20250731_012

+ freeze_pies
+ clearing_food_from_table_into_fridge, storing_food, Store_Fruit,
+ Open_And_Close_The_Refrigerator_Drawer_To_Place_Items, Organize_Refrigerator_Items, Dispose_Of_Leftover_Food, Organize_Refrigerated_shelf, Opening_Or_Closing_Refrigerator_Drawers_To_Store_And_Retrieve_Items
+ Organize_Refrigerated_Beverage_Case_20250704_003, Take_Drinks_From_The_Refrigerator_And_Place_Them_On_The_Table_20250728_010
