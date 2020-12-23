package DeliveryMain;


import java.util.ArrayList;
import java.util.List;

public class Menu {
     private List<Meal> dishlist = new ArrayList<>();

     public Menu() {}

     public List<Meal> getDishlist() {
         return dishlist;
     }

     public void setDishlist(List<Meal> dishlist) {
         this.dishlist = dishlist;
     }

     public void addDishToMenu(Meal meal){
         this.dishlist.add(meal);
     }
 }
