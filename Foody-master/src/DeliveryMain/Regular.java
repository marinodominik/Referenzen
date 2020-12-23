package DeliveryMain;


public class Regular extends Membership {
    private final String type = "regular";
    private final double fee = 25;

    public Regular( ) {}

    public String getType() {
        return type;
    }


    public double getFee() {
        return fee;
    }


}
