package DeliveryMain;

public class DisplayOnly extends Membership {
    private final String type = "display only";
    private final double fee = 10;

    public DisplayOnly() {}

    public String getType() {
        return type;
    }

    public double getFee() {
        return fee;
    }

}
