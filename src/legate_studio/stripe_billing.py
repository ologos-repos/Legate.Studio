"""
Stripe Billing Integration for Legato.Pit

Handles subscriptions, checkout, and webhooks for:
- BYOK tier ($0.99/month) - Bring your own API keys
- Managed tier ($10/month) - Platform API keys included
"""
import logging
import os
from functools import wraps

import stripe
from flask import Blueprint, request, jsonify, redirect, url_for, session, render_template, flash, current_app

from .core import login_required, get_effective_tier, get_trial_status

logger = logging.getLogger(__name__)

billing_bp = Blueprint('billing', __name__, url_prefix='/billing')

# Stripe product configuration
STRIPE_PRODUCTS = {
    'byok': {
        'name': 'Legate Studio BYOK',
        'description': 'Bring your own API keys - full features',
        'price_cents': 99,
        'interval': 'month',
        'tier': 'byok'
    },
    'managed': {
        'name': 'Legate Studio Managed',
        'description': 'Platform API keys included - full features',
        'price_cents': 1000,
        'interval': 'month',
        'tier': 'managed'
    }
}


def _get_db():
    """Get shared database for billing tables."""
    from .rag.database import init_db
    return init_db()


def _init_stripe():
    """Initialize Stripe with API key."""
    api_key = os.environ.get('STRIPE_SECRET_KEY')
    if not api_key:
        logger.warning("STRIPE_SECRET_KEY not set - billing disabled")
        return False
    stripe.api_key = api_key
    return True


def stripe_required(f):
    """Decorator to ensure Stripe is configured."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not _init_stripe():
            if request.is_json:
                return jsonify({'error': 'Billing not configured'}), 503
            flash('Billing is not currently available.', 'error')
            return redirect(url_for('dashboard.index'))
        return f(*args, **kwargs)
    return decorated


def get_or_create_stripe_products() -> dict:
    """Get or create Stripe products and prices.

    Returns dict mapping tier name to price_id.
    Stores IDs in system_config table.
    """
    db = _get_db()

    # Check if we already have product IDs stored
    config = db.execute(
        "SELECT key, value FROM system_config WHERE key LIKE 'stripe_price_%'"
    ).fetchall()

    if config and len(config) >= len(STRIPE_PRODUCTS):
        return {row['key'].replace('stripe_price_', ''): row['value'] for row in config}

    # Create products and prices in Stripe
    price_ids = {}

    for tier_key, product_config in STRIPE_PRODUCTS.items():
        try:
            # Check if product already exists by metadata
            existing = stripe.Product.search(
                query=f"metadata['legato_tier']:'{tier_key}'"
            )

            if existing.data:
                product = existing.data[0]
                logger.info(f"Found existing Stripe product for {tier_key}: {product.id}")
            else:
                # Create new product
                product = stripe.Product.create(
                    name=product_config['name'],
                    description=product_config['description'],
                    metadata={'legato_tier': tier_key}
                )
                logger.info(f"Created Stripe product for {tier_key}: {product.id}")

            # Get or create price
            prices = stripe.Price.list(product=product.id, active=True)
            if prices.data:
                price = prices.data[0]
            else:
                price = stripe.Price.create(
                    product=product.id,
                    unit_amount=product_config['price_cents'],
                    currency='usd',
                    recurring={'interval': product_config['interval']}
                )
                logger.info(f"Created Stripe price for {tier_key}: {price.id}")

            price_ids[tier_key] = price.id

            # Store in system_config
            db.execute("""
                INSERT OR REPLACE INTO system_config (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (f'stripe_price_{tier_key}', price.id))

        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating product {tier_key}: {e}")
            raise

    db.commit()
    return price_ids


def get_or_create_customer(user_id: str) -> str:
    """Get or create a Stripe customer for a user.

    Returns the Stripe customer ID.
    """
    db = _get_db()

    # Check if user already has a Stripe customer ID
    user = db.execute(
        "SELECT stripe_customer_id, github_login, email FROM users WHERE user_id = ?",
        (user_id,)
    ).fetchone()

    if not user:
        raise ValueError(f"User not found: {user_id}")

    if user['stripe_customer_id']:
        return user['stripe_customer_id']

    # Create new Stripe customer
    customer = stripe.Customer.create(
        email=user['email'],
        name=user['github_login'],
        metadata={'user_id': user_id, 'github_login': user['github_login']}
    )

    # Store customer ID
    db.execute(
        "UPDATE users SET stripe_customer_id = ? WHERE user_id = ?",
        (customer.id, user_id)
    )
    db.commit()

    logger.info(f"Created Stripe customer {customer.id} for user {user['github_login']}")
    return customer.id


def create_checkout_session(user_id: str, tier: str) -> str:
    """Create a Stripe checkout session for subscription.

    Args:
        user_id: User's ID
        tier: 'byok' or 'managed'

    Returns:
        Checkout session URL
    """
    if tier not in STRIPE_PRODUCTS:
        raise ValueError(f"Invalid tier: {tier}")

    price_ids = get_or_create_stripe_products()
    price_id = price_ids.get(tier)

    if not price_id:
        raise ValueError(f"No price found for tier: {tier}")

    customer_id = get_or_create_customer(user_id)

    # Get base URL for redirects
    base_url = request.host_url.rstrip('/')

    session = stripe.checkout.Session.create(
        customer=customer_id,
        payment_method_types=['card'],
        line_items=[{
            'price': price_id,
            'quantity': 1
        }],
        mode='subscription',
        success_url=f"{base_url}/billing/success?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{base_url}/billing/",
        metadata={
            'user_id': user_id,
            'tier': tier
        }
    )

    logger.info(f"Created checkout session {session.id} for user {user_id}, tier {tier}")
    return session.url


def create_portal_session(user_id: str) -> str:
    """Create a Stripe customer portal session.

    Returns portal URL for managing subscription.
    """
    customer_id = get_or_create_customer(user_id)
    base_url = request.host_url.rstrip('/')

    session = stripe.billing_portal.Session.create(
        customer=customer_id,
        return_url=f"{base_url}/billing/"
    )

    return session.url


def switch_subscription_tier(user_id: str, new_tier: str) -> dict:
    """Switch an existing subscription to a different tier.

    Args:
        user_id: User's ID
        new_tier: 'byok' or 'managed'

    Returns:
        Dict with status and message
    """
    if new_tier not in STRIPE_PRODUCTS:
        raise ValueError(f"Invalid tier: {new_tier}")

    db = _get_db()
    user = db.execute(
        "SELECT stripe_subscription_id, tier FROM users WHERE user_id = ?",
        (user_id,)
    ).fetchone()

    if not user or not user['stripe_subscription_id']:
        raise ValueError("No active subscription to switch")

    if user['tier'] == new_tier:
        raise ValueError(f"Already on {new_tier} tier")

    # Get price IDs
    price_ids = get_or_create_stripe_products()
    new_price_id = price_ids.get(new_tier)

    if not new_price_id:
        raise ValueError(f"No price found for tier: {new_tier}")

    # Get the subscription
    subscription = stripe.Subscription.retrieve(user['stripe_subscription_id'])

    # Update the subscription with the new price
    # This will prorate by default
    updated_subscription = stripe.Subscription.modify(
        user['stripe_subscription_id'],
        items=[{
            'id': subscription['items']['data'][0]['id'],
            'price': new_price_id,
        }],
        proration_behavior='create_prorations'
    )

    # Update local tier immediately (webhook will also update, but this is faster for UX)
    db.execute(
        "UPDATE users SET tier = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
        (new_tier, user_id)
    )
    db.commit()

    logger.info(f"Switched user {user_id} from {user['tier']} to {new_tier}")

    return {
        'success': True,
        'message': f'Switched to {new_tier} plan',
        'new_tier': new_tier
    }


def cancel_subscription(user_id: str) -> dict:
    """Cancel a user's subscription at period end.

    Args:
        user_id: User's ID

    Returns:
        Dict with status and message
    """
    db = _get_db()
    user = db.execute(
        "SELECT stripe_subscription_id, tier FROM users WHERE user_id = ?",
        (user_id,)
    ).fetchone()

    if not user or not user['stripe_subscription_id']:
        raise ValueError("No active subscription to cancel")

    # Cancel at period end (user keeps access until billing period ends)
    subscription = stripe.Subscription.modify(
        user['stripe_subscription_id'],
        cancel_at_period_end=True
    )

    logger.info(f"Scheduled cancellation for user {user_id} subscription")

    return {
        'success': True,
        'message': 'Subscription will be cancelled at the end of the billing period',
        'cancel_at': subscription.get('cancel_at')
    }


def reactivate_subscription(user_id: str) -> dict:
    """Reactivate a subscription that was scheduled for cancellation.

    Args:
        user_id: User's ID

    Returns:
        Dict with status and message
    """
    db = _get_db()
    user = db.execute(
        "SELECT stripe_subscription_id FROM users WHERE user_id = ?",
        (user_id,)
    ).fetchone()

    if not user or not user['stripe_subscription_id']:
        raise ValueError("No subscription to reactivate")

    # Remove the cancellation
    subscription = stripe.Subscription.modify(
        user['stripe_subscription_id'],
        cancel_at_period_end=False
    )

    logger.info(f"Reactivated subscription for user {user_id}")

    return {
        'success': True,
        'message': 'Subscription reactivated'
    }


# ============ Webhook Handlers ============

def handle_checkout_completed(session_data: dict):
    """Handle successful checkout - activate subscription."""
    user_id = session_data.get('metadata', {}).get('user_id')
    tier = session_data.get('metadata', {}).get('tier')
    subscription_id = session_data.get('subscription')
    customer_id = session_data.get('customer')

    if not user_id or not tier:
        logger.warning(f"Checkout completed but missing metadata: {session_data}")
        return

    db = _get_db()

    # Check is_beta FIRST — beta users are immune to webhook tier changes (admin-managed)
    user = db.execute(
        "SELECT github_login, is_beta FROM users WHERE user_id = ?",
        (user_id,)
    ).fetchone()
    if user and user['is_beta']:
        logger.info(
            f"Skipping checkout tier activation for beta user {user['github_login']} "
            f"— beta tier is admin-managed. Still recording Stripe IDs."
        )
        # Still record customer/subscription IDs even for beta users — useful for admin visibility
        db.execute("""
            UPDATE users SET
                stripe_subscription_id = COALESCE(stripe_subscription_id, ?),
                stripe_customer_id = COALESCE(stripe_customer_id, ?),
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ?
        """, (subscription_id, customer_id, user_id))
        db.commit()
        return

    # Update user's tier and subscription
    db.execute("""
        UPDATE users SET
            tier = ?,
            stripe_subscription_id = ?,
            stripe_customer_id = COALESCE(stripe_customer_id, ?),
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = ?
    """, (tier, subscription_id, customer_id, user_id))
    db.commit()

    logger.info(f"Activated {tier} subscription for user {user_id}")


def handle_subscription_updated(subscription: dict):
    """Handle subscription updates (upgrade/downgrade)."""
    subscription_id = subscription.get('id')
    customer_id = subscription.get('customer')
    status = subscription.get('status')

    db = _get_db()

    # Find user by subscription or customer ID — fetch is_beta and current tier too
    user = db.execute(
        "SELECT user_id, github_login, tier, is_beta FROM users WHERE stripe_subscription_id = ? OR stripe_customer_id = ?",
        (subscription_id, customer_id)
    ).fetchone()

    if not user:
        logger.warning(f"Subscription updated but no user found: {subscription_id}")
        return

    # Beta users are immune to webhook tier changes — their tier is managed by admins
    if user['is_beta']:
        logger.info(f"Skipping subscription update for beta user {user['github_login']} — beta tier is admin-managed")
        return

    # Get the price/tier from subscription items
    items = subscription.get('items', {}).get('data', [])
    if items:
        price_id = items[0].get('price', {}).get('id')
        try:
            # Look up tier by price ID in system_config
            config = db.execute(
                "SELECT key FROM system_config WHERE value = ?",
                (price_id,)
            ).fetchone()
        except Exception as e:
            logger.warning(f"system_config lookup failed (table may not be initialized): {e}")
            config = None

        if config:
            tier = config['key'].replace('stripe_price_', '')
        else:
            # system_config lookup failed or price not found — PRESERVE existing tier, do not overwrite
            logger.warning(
                f"Could not resolve tier for price_id={price_id} (subscription {subscription_id}). "
                f"Preserving existing tier='{user['tier']}' for user {user['github_login']}."
            )
            tier = user['tier']
    else:
        # No items in subscription — preserve existing tier
        logger.warning(
            f"Subscription {subscription_id} has no items. "
            f"Preserving existing tier='{user['tier']}' for user {user['github_login']}."
        )
        tier = user['tier']

    # Update tier based on subscription status
    if status in ('active', 'trialing'):
        db.execute(
            "UPDATE users SET tier = ?, stripe_subscription_id = ? WHERE user_id = ?",
            (tier, subscription_id, user['user_id'])
        )
    elif status in ('past_due', 'unpaid'):
        # Keep tier but flag payment issue
        logger.warning(f"Subscription {subscription_id} is {status} for user {user['github_login']}")
    elif status in ('canceled', 'incomplete_expired'):
        db.execute(
            "UPDATE users SET tier = 'trial', stripe_subscription_id = NULL WHERE user_id = ?",
            (user['user_id'],)
        )

    db.commit()
    logger.info(f"Updated subscription for user {user['github_login']}: status={status}, tier={tier}")


def handle_subscription_deleted(subscription: dict):
    """Handle subscription cancellation."""
    subscription_id = subscription.get('id')

    db = _get_db()

    user = db.execute(
        "SELECT user_id, github_login, tier, is_beta FROM users WHERE stripe_subscription_id = ?",
        (subscription_id,)
    ).fetchone()

    if not user:
        logger.warning(f"Subscription deleted but no user found: {subscription_id}")
        return

    # Beta users are immune to webhook tier changes — admin-managed, never downgrade via webhook
    if user['is_beta']:
        logger.info(f"Skipping subscription deletion downgrade for beta user {user['github_login']} — beta tier is admin-managed")
        # Still clear the subscription ID so we don't track a dead sub, but preserve tier
        db.execute("""
            UPDATE users SET
                stripe_subscription_id = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ?
        """, (user['user_id'],))
        db.commit()
        return

    # Revert to trial tier for non-beta users
    db.execute("""
        UPDATE users SET
            tier = 'trial',
            stripe_subscription_id = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = ?
    """, (user['user_id'],))
    db.commit()

    logger.info(f"Subscription cancelled for user {user['github_login']}, reverted to trial")


def handle_invoice_payment_failed(invoice: dict):
    """Handle failed payment - nag but don't block."""
    customer_id = invoice.get('customer')

    db = _get_db()

    user = db.execute(
        "SELECT user_id, github_login FROM users WHERE stripe_customer_id = ?",
        (customer_id,)
    ).fetchone()

    if user:
        logger.warning(f"Payment failed for user {user['github_login']}")
        # Could set a flag here for UI nag, but for now just log


# ============ Routes ============

@billing_bp.route('/')
@login_required
def index():
    """Billing overview page."""
    user_id = session['user']['user_id']
    tier = get_effective_tier(user_id)
    trial_status = get_trial_status(user_id)

    db = _get_db()
    user = db.execute(
        "SELECT tier, is_beta, stripe_customer_id, stripe_subscription_id FROM users WHERE user_id = ?",
        (user_id,)
    ).fetchone()

    # Check if subscription is scheduled for cancellation
    cancel_at_period_end = False
    period_end_date = None
    if user and user['stripe_subscription_id'] and _init_stripe():
        try:
            subscription = stripe.Subscription.retrieve(user['stripe_subscription_id'])
            cancel_at_period_end = subscription.get('cancel_at_period_end', False)
            current_period_end = subscription.get('current_period_end')
            if current_period_end:
                from datetime import datetime
                period_end_date = datetime.fromtimestamp(current_period_end).strftime('%B %d, %Y')
        except Exception as e:
            logger.warning(f"Could not fetch subscription details: {e}")

    return render_template('billing.html',
        title='Billing',
        tier=tier,
        raw_tier=user['tier'] if user else 'trial',
        is_beta=user['is_beta'] if user else False,
        trial_status=trial_status,
        has_subscription=bool(user['stripe_subscription_id']) if user else False,
        cancel_at_period_end=cancel_at_period_end,
        period_end_date=period_end_date,
        stripe_enabled=bool(os.environ.get('STRIPE_SECRET_KEY')),
        products=STRIPE_PRODUCTS
    )


@billing_bp.route('/checkout', methods=['POST'])
@login_required
@stripe_required
def checkout():
    """Create checkout session and redirect to Stripe."""
    user_id = session['user']['user_id']
    tier = request.form.get('tier') or request.json.get('tier')

    if tier not in ('byok', 'managed'):
        if request.is_json:
            return jsonify({'error': 'Invalid tier'}), 400
        flash('Invalid subscription tier.', 'error')
        return redirect(url_for('billing.index'))

    try:
        checkout_url = create_checkout_session(user_id, tier)
        if request.is_json:
            return jsonify({'url': checkout_url})
        return redirect(checkout_url)
    except Exception as e:
        logger.error(f"Checkout error: {e}")
        if request.is_json:
            return jsonify({'error': str(e)}), 500
        flash(f'Error creating checkout: {e}', 'error')
        return redirect(url_for('billing.index'))


@billing_bp.route('/success')
@login_required
def success():
    """Checkout success page."""
    session_id = request.args.get('session_id')
    flash('Subscription activated! Thank you for subscribing.', 'success')
    return redirect(url_for('billing.index'))


@billing_bp.route('/portal', methods=['POST'])
@login_required
@stripe_required
def portal():
    """Redirect to Stripe customer portal."""
    user_id = session['user']['user_id']

    try:
        portal_url = create_portal_session(user_id)
        return redirect(portal_url)
    except Exception as e:
        logger.error(f"Portal error: {e}")
        flash(f'Error accessing billing portal: {e}', 'error')
        return redirect(url_for('billing.index'))


@billing_bp.route('/switch', methods=['POST'])
@login_required
@stripe_required
def switch_tier():
    """Switch subscription to a different tier."""
    user_id = session['user']['user_id']
    new_tier = request.form.get('tier') or request.json.get('tier')

    if new_tier not in ('byok', 'managed'):
        if request.is_json:
            return jsonify({'error': 'Invalid tier'}), 400
        flash('Invalid subscription tier.', 'error')
        return redirect(url_for('billing.index'))

    try:
        result = switch_subscription_tier(user_id, new_tier)
        if request.is_json:
            return jsonify(result)
        flash(f'Successfully switched to {new_tier.upper()} plan!', 'success')
        return redirect(url_for('billing.index'))
    except ValueError as e:
        if request.is_json:
            return jsonify({'error': str(e)}), 400
        flash(str(e), 'error')
        return redirect(url_for('billing.index'))
    except Exception as e:
        logger.error(f"Switch tier error: {e}")
        if request.is_json:
            return jsonify({'error': str(e)}), 500
        flash(f'Error switching plan: {e}', 'error')
        return redirect(url_for('billing.index'))


@billing_bp.route('/cancel', methods=['POST'])
@login_required
@stripe_required
def cancel():
    """Cancel subscription at end of billing period."""
    user_id = session['user']['user_id']

    try:
        result = cancel_subscription(user_id)
        if request.is_json:
            return jsonify(result)
        flash('Your subscription will be cancelled at the end of the billing period.', 'info')
        return redirect(url_for('billing.index'))
    except ValueError as e:
        if request.is_json:
            return jsonify({'error': str(e)}), 400
        flash(str(e), 'error')
        return redirect(url_for('billing.index'))
    except Exception as e:
        logger.error(f"Cancel error: {e}")
        if request.is_json:
            return jsonify({'error': str(e)}), 500
        flash(f'Error cancelling subscription: {e}', 'error')
        return redirect(url_for('billing.index'))


@billing_bp.route('/reactivate', methods=['POST'])
@login_required
@stripe_required
def reactivate():
    """Reactivate a subscription scheduled for cancellation."""
    user_id = session['user']['user_id']

    try:
        result = reactivate_subscription(user_id)
        if request.is_json:
            return jsonify(result)
        flash('Your subscription has been reactivated!', 'success')
        return redirect(url_for('billing.index'))
    except ValueError as e:
        if request.is_json:
            return jsonify({'error': str(e)}), 400
        flash(str(e), 'error')
        return redirect(url_for('billing.index'))
    except Exception as e:
        logger.error(f"Reactivate error: {e}")
        if request.is_json:
            return jsonify({'error': str(e)}), 500
        flash(f'Error reactivating subscription: {e}', 'error')
        return redirect(url_for('billing.index'))


@billing_bp.route('/webhook', methods=['POST'])
def webhook():
    """Handle Stripe webhooks."""
    payload = request.get_data()
    sig_header = request.headers.get('Stripe-Signature')
    webhook_secret = os.environ.get('STRIPE_WEBHOOK_SECRET')

    if not webhook_secret:
        logger.error("STRIPE_WEBHOOK_SECRET not configured")
        return jsonify({'error': 'Webhook not configured'}), 500

    try:
        _init_stripe()
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        )
    except ValueError:
        logger.error("Invalid webhook payload")
        return jsonify({'error': 'Invalid payload'}), 400
    except stripe.error.SignatureVerificationError:
        logger.error("Invalid webhook signature")
        return jsonify({'error': 'Invalid signature'}), 400

    # Handle event types
    event_type = event['type']
    data = event['data']['object']

    logger.info(f"Received Stripe webhook: {event_type}")

    if event_type == 'checkout.session.completed':
        handle_checkout_completed(data)
    elif event_type == 'customer.subscription.updated':
        handle_subscription_updated(data)
    elif event_type == 'customer.subscription.deleted':
        handle_subscription_deleted(data)
    elif event_type == 'invoice.payment_failed':
        handle_invoice_payment_failed(data)
    else:
        logger.debug(f"Unhandled webhook event type: {event_type}")

    return jsonify({'received': True})


# ============ Initialization ============

def init_stripe_products_on_startup(app):
    """Initialize Stripe products on app startup.

    Call this from create_app() to ensure products exist.
    """
    if not os.environ.get('STRIPE_SECRET_KEY'):
        logger.info("Stripe not configured - skipping product initialization")
        return

    try:
        _init_stripe()
        with app.app_context():
            products = get_or_create_stripe_products()
            logger.info(f"Stripe products initialized: {list(products.keys())}")
    except Exception as e:
        logger.error(f"Failed to initialize Stripe products: {e}")
