package DeliveryMain;


public class Premium extends Membership {
    private final String type = "premium";
    private final double fee = 49;

    public Premium() {}

    public String getType() {
        return type;
    }

    public double getFee() {
        return fee;
    }

}
